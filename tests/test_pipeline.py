"""濃度計算・処理履歴・QC・外部データ結合・設定の保存と再読込の検証 (模擬データ)。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lcms_analysis as la  # noqa: E402
import lcms_external as lx  # noqa: E402
import lcms_pipeline as lp  # noqa: E402
import lcms_qc as qc  # noqa: E402
from synthetic import make_text  # noqa: E402

CPDS = ["IS", "AA1", "AA2", "AA3"]


@pytest.fixture(scope="module")
def data():
    raw_bytes = make_text()
    raw, is_df, cpds = la.load_tables(raw_bytes, "IS")
    full = qc.parse_full(raw_bytes)
    rng = qc.calibration_ranges(full).reindex(cpds)
    rng["一点検量"] = rng["一点検量"].fillna(True).astype(bool)
    return full, raw, cpds, rng


def val(res, row, cpd):
    p = res.provenance
    return p[(p["行"] == row) & (p["化合物"] == cpd)].iloc[0]


# ---------------------------------------------------------------- IS の二重補正の防止
def test_is_not_applied_when_method_unset(data):
    full, raw, cpds, rng = data
    res = lp.run(full, raw, cpds, lp.QuantSettings(app_is=True, is_name="IS"), rng)
    r = val(res, "20260101_S-2_2_012.lcd", "AA2")
    assert r["IS 係数"] == 1.0 and r["最終値"] == pytest.approx(6.0)
    assert any("未設定" in n for n in res.notes)


def test_is_not_applied_for_internal_standard_method(data):
    full, raw, cpds, rng = data
    s = lp.QuantSettings(quant_method=lp.QM_INTERNAL, app_is=True, is_name="IS")
    res = lp.run(full, raw, cpds, s, rng)
    assert val(res, "20260101_S-2_2_012.lcd", "AA2")["IS 係数"] == 1.0
    assert any("二重補正" in n for n in res.notes)


def test_is_applied_once_for_external_method(data):
    full, raw, cpds, rng = data
    s = lp.QuantSettings(quant_method=lp.QM_EXTERNAL, app_is=True, is_name="IS")
    res = lp.run(full, raw, cpds, s, rng)
    # 希釈なし群の IS 平均 = (10 + 20 + 10 + 10) / 4 = 12.5 (再注入は最初の測定のみ使う設定でも IS 平均は全ファイルから)
    is_mean = raw[(raw["希釈倍率"] == "") & ~raw["label"].str.contains("STD")]["IS"].mean()
    r = val(res, "20260101_S-2_2_012.lcd", "AA2")
    assert r["IS 係数"] == pytest.approx(is_mean / 20.0)
    assert r["最終値"] == pytest.approx(6.0 * is_mean / 20.0)


# ---------------------------------------------------------------- 希釈係数・体積係数の重複防止
def test_dilution_not_applied_twice_when_labsolutions_applied(data):
    full, raw, cpds, rng = data
    s = lp.QuantSettings(dilution_state=lp.DIL_APPLIED)
    res = lp.run(full, raw, cpds, s, rng, merge=True)
    r = val(res, "S-1", "AA1")  # 希釈なしが上限超過 -> 希釈測定を採用。倍率は掛けない
    assert r["希釈"] == "x10" and r["希釈係数"] == 1.0 and r["最終値"] == pytest.approx(8.2)


def test_dilution_and_volume_applied_once(data):
    full, raw, cpds, rng = data
    s = lp.QuantSettings(dilution_state=lp.DIL_NOT_APPLIED, volume_factor=5.0)
    res = lp.run(full, raw, cpds, s, rng, merge=True)
    r = val(res, "S-1", "AA1")
    assert r["希釈係数"] == 10.0 and r["体積換算係数"] == 5.0 and r["最終値"] == pytest.approx(8.2 * 10 * 5)
    # 自動 (化合物単位): 1 試料でも上限を超えたら、その化合物は全試料で希釈測定を使う (希釈間の混在を避ける)
    r3 = val(res, "S-3", "AA1")
    assert r3["希釈"] == "x10" and r3["最終値"] == pytest.approx(1.1 * 10 * 5)
    # 希釈なしを採用した値には希釈係数を掛けない
    res_u = lp.run(full, raw, cpds, s, rng, {"AA1": qc.SRC_NONE}, merge=True)
    r3u = val(res_u, "S-3", "AA1")
    assert r3u["希釈"] == "希釈なし" and r3u["希釈係数"] == 1.0 and r3u["最終値"] == pytest.approx(10.0 * 5)


def test_merge_refused_when_dilution_state_unknown(data):
    full, raw, cpds, rng = data
    res = lp.run(full, raw, cpds, lp.QuantSettings(), rng, merge=True)
    assert "統合" not in set(res.values["希釈倍率"])  # 希釈グループ別のまま
    assert any("希釈の統合には" in n for n in res.notes)


def test_mult_dilution_by_group_respects_state(data):
    full, raw, cpds, rng = data
    applied = lp.run(full, raw, cpds, lp.QuantSettings(dilution_state=lp.DIL_APPLIED), rng, mult_dilution=True)
    assert val(applied, "20260101_S-1 x10_1_011.lcd", "AA1")["最終値"] == pytest.approx(8.2)
    not_applied = lp.run(full, raw, cpds, lp.QuantSettings(dilution_state=lp.DIL_NOT_APPLIED), rng, mult_dilution=True)
    assert val(not_applied, "20260101_S-1 x10_1_011.lcd", "AA1")["最終値"] == pytest.approx(82.0)


# ---------------------------------------------------------------- 検量範囲外の値を採用しない
def test_out_of_range_not_adopted(data):
    full, raw, cpds, rng = data
    res = lp.run(full, raw, cpds, lp.QuantSettings(dilution_state=lp.DIL_NOT_APPLIED), rng, merge=True)
    r = val(res, "S-2", "AA1")  # 900 (希釈なし) と 90 (x10) がどちらも上限 50 を超える
    assert r["状態"] == lp.ST_REMEASURE and np.isnan(r["最終値"])
    assert "900" in r["採用しなかった測定"] and "90" in r["採用しなかった測定"]
    v = res.values.set_index("label").at["S-2", "AA1"]
    assert np.isnan(v)


def test_single_point_calibration_is_not_evaluated(data):
    full, raw, cpds, rng = data
    res = lp.run(full, raw, cpds, lp.QuantSettings(), rng)
    r = val(res, "20260101_S-1_1_010.lcd", "AA2")
    assert r["状態"] == lp.ST_CHECK and "未評価" in r["理由"]


def test_input_error_and_exclusion(data):
    full, raw, cpds, rng = data
    res = lp.run(full, raw, cpds, lp.QuantSettings(), rng)
    r = val(res, "20260101_S-3_3_014.lcd", "AA3")
    assert r["状態"] == lp.ST_ERR and np.isnan(r["最終値"])


def test_dilution_consistency_flag(data):
    full, raw, cpds, rng = data
    s = lp.QuantSettings(dilution_state=lp.DIL_NOT_APPLIED, consistency_tol=5)
    res = lp.run(full, raw, cpds, s, rng, merge=True)
    r = val(res, "S-3", "AA1")  # 希釈なし 10、x10 で 1.1 x 10 = 11 (10% の差 > 5%)
    assert r["状態"] == lp.ST_CHECK and "不一致" in r["理由"]


def test_reinjection_first_or_mean(data):
    full, raw, cpds, rng = data
    first = lp.run(full, raw, cpds, lp.QuantSettings(), rng)
    s4 = first.values[(first.values["label"] == "S-4") & (first.values["希釈倍率"] == "")]
    assert len(s4) == 1 and s4["AA1"].iloc[0] == pytest.approx(20.0)
    assert (first.provenance["行"] == "(不採用)").any()
    mean = lp.run(full, raw, cpds, lp.QuantSettings(reinjection=lp.REINJECT_MEAN), rng)
    s4m = mean.values[(mean.values["label"] == "S-4") & (mean.values["希釈倍率"] == "")]
    assert len(s4m) == 1 and s4m["AA1"].iloc[0] == pytest.approx(25.0)


def test_mask_keeps_original_value(data):
    full, raw, cpds, rng = data
    res = lp.run(full, raw, cpds, lp.QuantSettings(), rng, mask_level=qc.MASK_LOQ, mask_repl=qc.REPL_HALF)
    assert (res.provenance["マスク"] == "").all()  # 模擬データに LOQ 未満は無い
    assert "入力値" in res.provenance  # 元の値は常に残る


# ---------------------------------------------------------------- 技術反復と生物学的 n
def test_technical_replicates_are_not_counted_as_biological_n():
    base = pd.DataFrame({"データファイル名": ["f1", "f2", "f3", "f4"], "label": ["A1", "A1b", "B1", "QC1"],
                         "希釈倍率": "", "condition": ["C", "C", "C", ""], "希釈": "希釈なし",
                         "X": [1.0, 3.0, 5.0, 9.0]})
    meta = pd.DataFrame({lp.TYPE_COL: ["試料", "試料", "試料", "プール QC"], lp.SUBJECT_COL: ["m1", "m1", "m2", "QC1"],
                         lp.TIME_COL: "", "実験回": "", "前処理バッチ": "", "測定バッチ": ""},
                        index=["A1", "A1b", "B1", "QC1"])
    out, info = lp.apply_metadata(base, meta, ["X"])
    assert len(out) == 2  # m1 (技術反復 2 行を平均) と m2。QC は除く
    assert out.set_index(lp.SUBJECT_COL).at["m1", "X"] == pytest.approx(2.0)
    assert "生物学的 n (個体数): 希釈なし: 2" in info


# ---------------------------------------------------------------- 外部データの結合
def test_external_merge_does_not_multiply_rows():
    ext = pd.DataFrame({"ID": ["S-1", "S-1", "S-2"], "TG": [1.0, 3.0, 5.0]})
    wide, dup = lx.prepare(ext, "ID", ["TG"], "lip", lx.MATCH_EXACT)
    assert dup == 1 and wide.loc["s-1", "lip: TG"] == pytest.approx(2.0)
    base = pd.DataFrame({"label": ["S-1", "S-1", "S-2", "S-3"], "希釈": ["希釈なし", "x10", "希釈なし", "希釈なし"]})
    out, matched, unmatched = lx.merge_into(base, "label", wide, lx.MATCH_EXACT)
    assert len(out) == len(base)
    assert out["lip: TG"].isna().sum() == 1  # S-3 は外部データに無い


# ---------------------------------------------------------------- QC
def test_qc_not_evaluated_without_qc(data):
    full, raw, cpds, rng = data
    ev = qc.qc_evaluation(full, {}, qc.injection_order(full))
    assert (ev["プール QC 判定"] == qc.NOT_EVALUATED).all()
    assert (ev["既知濃度 QC 判定"].str.startswith(qc.NOT_EVALUATED)).all()


def test_pool_qc_cv(data):
    full, raw, cpds, rng = data
    types = {"S-1": "プール QC", "S-2": "プール QC", "S-3": "プール QC"}
    ev = qc.qc_evaluation(full, types, qc.injection_order(full)).set_index("化合物")
    assert ev.at["AA2", "プール QC n"] == 6  # 希釈なしと x10 の両方
    assert ev.at["AA2", "プール QC 判定"] in ("合格", "不合格")


def test_calibration_lloq(data):
    full, raw, cpds, rng = data
    cal = qc.calibration_detail(full).set_index("化合物")
    assert cal.at["AA1", "確認済み LLOQ"] == "1"  # 1 は 3 回測定 (偏り +0.7%, CV 約 4%)
    assert cal.at["AA2", "確認済み LLOQ"] == "未確認"  # 1 点検量・繰り返しなし
    assert "未取得" in cal.at["AA1", "検量線の式・重み付け"]


# ---------------------------------------------------------------- 設定の保存と再読込
def test_settings_roundtrip(monkeypatch):
    import ui_project

    state = {"q_method": lp.QM_EXTERNAL, "q_app_is": True, "q_vol": 10.0, "_btn_x": False, "data_pick": Path("data/x.txt"),
             "cond_cur::ds": pd.DataFrame({"label": ["S-1"], "condition": ["A"], "使用": [True]})}
    monkeypatch.setattr(ui_project.st, "session_state", state)
    proj = ui_project.snapshot([("ds", b"abc")], [("lipid.csv", b"xyz")], {"アプリの版": "test"})
    assert proj["data"][0]["sha256"] == ui_project.sha256(b"abc")
    assert proj["external"][0]["name"] == "lipid.csv" and proj["run_info"]["アプリの版"] == "test"
    assert "_btn_x" not in proj["widgets"]  # ボタンは保存しない
    restored = {}
    monkeypatch.setattr(ui_project.st, "session_state", restored)
    ui_project.apply(json.loads(json.dumps(proj)), Path("data"))
    assert restored["q_method"] == lp.QM_EXTERNAL and restored["q_vol"] == 10.0
    assert restored["data_pick"] == Path("data/x.txt")
    assert restored["cond_init::ds"]["condition"].tolist() == ["A"]


# ---------------------------------------------------------------- 換算濃度 (バイアル中濃度 -> 元の血清中濃度)
def test_conversion_factor_presets():
    const = lp.CONV_CONST_DEFAULT
    assert lp.conversion_factor(const, 50) == pytest.approx(200 * 380 / 280 * 170 / 130 / 50)  # ラット
    assert lp.conversion_factor(const, 25) == pytest.approx(200 * 380 / 280 * 170 / 130 / 25)  # マウス
    with pytest.raises(ValueError):
        lp.eval_expression("__import__('os')")


def test_converted_concentration_with_per_sample_override(data):
    full, raw, cpds, rng = data
    rat = lp.conversion_factor(lp.CONV_CONST_DEFAULT, 50)
    mouse = lp.conversion_factor(lp.CONV_CONST_DEFAULT, 25)
    s = lp.QuantSettings(dilution_state=lp.DIL_NOT_APPLIED, volume_factor=rat, volume_by_label={"S-3": mouse})
    res = lp.run(full, raw, cpds, s, rng)
    r1 = val(res, "20260101_S-1_1_010.lcd", "AA2")
    assert r1["最終値"] == pytest.approx(5.0 * rat) and r1["最終値の意味"] == "換算濃度"
    r3 = val(res, "20260101_S-3_3_014.lcd", "AA2")
    assert r3["最終値"] == pytest.approx(7.0 * mouse)
    # 換算濃度にするときは、希釈測定の倍率も換算に含める (LabSolutions で未適用と設定した場合)
    rx = val(res, "20260101_S-1 x10_1_011.lcd", "AA2")
    assert rx["希釈係数"] == 10.0 and rx["最終値"] == pytest.approx(0.5 * 10 * rat)


def test_converted_concentration_flags_unconverted_dilution(data):
    full, raw, cpds, rng = data
    s = lp.QuantSettings(volume_factor=7.0)  # 希釈測定の倍率は未設定
    res = lp.run(full, raw, cpds, s, rng)
    rx = val(res, "20260101_S-1 x10_1_011.lcd", "AA2")
    assert rx["希釈係数"] == 1.0 and "換算濃度ではない" in rx["最終値の意味"]
    assert any("換算濃度になっていません" in n for n in res.notes)
