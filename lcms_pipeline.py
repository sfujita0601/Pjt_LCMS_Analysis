"""濃度の計算と処理履歴。

LabSolutions の出力値 (測定液中の濃度) から、解析に使う値を次の順で作り、値ごとに経過を記録する。

    1. 入力値          LabSolutions の「濃度」(未検出は「-----」)
    2. 状態の判定      入力エラー / 測定なし / 未検出 / LOD 未満 / LOQ 未満 / 検量上限超過 / 検量下限未満 / 確認必要 / 採用可能
    3. 測定の採用      希釈を統合する場合: 化合物ごとに 希釈なし / 希釈測定 のどちらを採用するか、採用しない (再測定候補) か
    4. IS 係数         アプリでの内部標準補正 (定量方式が「外部標準法」と設定され、補正を有効にした場合のみ)
    5. 希釈係数        希釈測定の倍率 (LabSolutions で未適用と設定した場合のみ掛ける)
    6. 体積換算係数    前処理による換算 (設定した場合のみ。最終値は「元の血清・血漿中濃度」になる)
    7. マスク          LOD / LOQ 未満の値の置き換え (設定した場合のみ)

不明な項目 (定量方式・希釈係数の適用状態・体積換算) は推測で補わず、未設定のまま係数 1 として扱い、そのことを表示する。
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import lcms_qc as qc
from lcms_analysis import COND_COL, GROUP_COL
from make_conc_table import DILUTION_COL, FILE_COL, LABEL_COL, is_std

# ---------------------------------------------------------------- 設定
QM_UNSET = "未設定"
QM_EXTERNAL = "外部標準法 (LabSolutions で IS 補正なし)"
QM_INTERNAL = "内部標準法 (LabSolutions で IS 面積比を使用済み)"
QUANT_METHODS = [QM_UNSET, QM_EXTERNAL, QM_INTERNAL]

DIL_UNSET = "未設定"
DIL_NOT_APPLIED = "未適用 (アプリで希釈倍率を掛ける)"
DIL_APPLIED = "LabSolutions で適用済み"
DILUTION_STATES = [DIL_UNSET, DIL_NOT_APPLIED, DIL_APPLIED]

IS_STAGES = ["未設定", "前処理の前 (試料に添加)", "前処理の後 (抽出液に添加)", "測定の直前", "その他"]

# 換算濃度 (バイアル中濃度 -> 元の血清中濃度) の式: 換算係数 = 定数部分 x 1 / 出発血清量 (µL)
# 定数部分は前処理の体積比 (200 x 380/280 x 170/130)。出発血清量は動物種で異なる
CONV_CONST_DEFAULT = "200 * 380 / 280 * 170 / 130"
SERUM_PRESETS = {
    "ラット (血清 50 µL)": 50.0,
    "マウス (血清 25 µL + 水 25 µL)": 25.0,
    "カスタム": None,
}


def eval_expression(text):
    """四則演算だけの式を評価する (例: "200 * 380 / 280")。それ以外の構文はエラーにする。"""
    import ast
    import operator

    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("数値と + - * / ( ) だけが使えます")

    return ev(ast.parse(str(text), mode="eval"))


def conversion_factor(const_expr, serum_ul):
    """換算係数 = 定数部分 / 出発血清量。"""
    return eval_expression(const_expr) / float(serum_ul)
REINJECT_FIRST, REINJECT_MEAN = "平均しない (最初に測定した値を採用)", "平均する"

# 値の状態
ST_OK, ST_CHECK = "採用可能", "確認必要"
ST_LOD, ST_LOQ = "LOD 未満", "LOQ 未満"
ST_OVER, ST_UNDER = "検量上限超過", "検量下限未満"
ST_NONE, ST_ND, ST_ERR = "測定なし", "未検出", "入力エラー"
ST_REMEASURE = "再測定候補"
STATUS_ORDER = [ST_OK, ST_CHECK, ST_UNDER, ST_LOQ, ST_LOD, ST_ND, ST_OVER, ST_REMEASURE, ST_NONE, ST_ERR]
# 採用してよい状態 (値があり、検量範囲を超えていない)。それ以外は採用しない
ADOPTABLE = {ST_OK, ST_CHECK, ST_UNDER, ST_LOQ, ST_LOD}


@dataclass
class QuantSettings:
    quant_method: str = QM_UNSET
    app_is: bool = False            # アプリで IS 補正を行うか (外部標準法のときだけ有効)
    is_name: str = ""
    is_stage: str = "未設定"
    is_conc_def: str = ""           # IS の濃度の定義 (自由記述)
    dilution_state: str = DIL_UNSET
    volume_factor: float | None = None  # 前処理による体積換算 (None = 換算しない)
    volume_def: str = ""
    volume_by_label: dict = field(default_factory=dict)  # 試料ごとの換算係数 (出発血清量を個別に指定した場合)
    unit: str = "未設定"            # LabSolutions の濃度の単位 (エクスポートに含まれないため利用者が設定)
    consistency_tol: float = 20.0   # 希釈間の一致の許容 (%)
    reinjection: str = REINJECT_FIRST

    def is_applied(self):
        return self.app_is and self.quant_method == QM_EXTERNAL and bool(self.is_name)

    def dilution_known(self):
        return self.dilution_state != DIL_UNSET

    def meaning(self, normalized=False):
        """最終値の意味 (表や図の説明に使う)。"""
        if normalized:
            return "正規化後の相対値 (濃度ではない)"
        base = "換算濃度 (元の血清中濃度)" if self.volume_factor else "バイアル中濃度 (測定液中濃度)"
        parts = [base]
        if self.is_applied():
            parts.append("IS 補正後")
        unit = f"単位: {self.unit}" if self.unit and self.unit != "未設定" else "単位: 未設定"
        return " / ".join(parts) + f" ({unit})"

    def vol_for(self, label):
        if not self.volume_factor:
            return 1.0
        return float(self.volume_by_label.get(label, self.volume_factor))

    def to_dict(self):
        return asdict(self)


def diagnose_quant_method(full, is_name):
    """LabSolutions の濃度が 面積 に比例するか (外部標準法)、面積 / IS 面積 に比例するか (内部標準法) を調べる。

    診断であって判定ではない (検量線に切片や重み付けがあると比は一定にならない)。
    戻り値: dict(n_compounds, cv_external, cv_internal, hint)
    """
    if not is_name or is_name not in set(full["compound"]):
        return dict(n_compounds=0, cv_external=np.nan, cv_internal=np.nan, hint="IS が未選択のため診断できません")
    is_area = full[full["compound"] == is_name].drop_duplicates("file").set_index("file")["area"]
    d = full[(full["compound"] != is_name) & full["conc"].notna() & (full["area"] > 0)].copy()
    d["k_ext"] = d["conc"] / d["area"]
    d["k_is"] = d["conc"] / (d["area"] / d["file"].map(is_area))
    g = d.groupby("compound").agg(n=("conc", "size"),
                                  cv_ext=("k_ext", lambda v: v.std() / v.mean() * 100),
                                  cv_is=("k_is", lambda v: v.std() / v.mean() * 100))
    g = g[g["n"] >= 5]
    if g.empty:
        return dict(n_compounds=0, cv_external=np.nan, cv_internal=np.nan, hint="判断に十分なデータがありません")
    ce, ci = float(g["cv_ext"].median()), float(g["cv_is"].median())
    if ce < 1 and ci > 5:
        hint = "濃度が面積にほぼ比例しています。外部標準法 (IS 補正なし) の可能性が高いと考えられます"
    elif ci < 1 and ce > 5:
        hint = "濃度が 面積 / IS 面積 にほぼ比例しています。内部標準法の可能性が高いと考えられます"
    else:
        hint = "どちらとも判断できません (検量線の切片・重み付けなどの影響の可能性)"
    return dict(n_compounds=int(len(g)), cv_external=ce, cv_internal=ci, hint=hint)


def is_factors(raw, is_name, used=None):
    """アプリでの IS 補正係数 (ファイルごと)。係数 = 希釈グループ内の IS 平均 / そのファイルの IS。

    STD と、解析に使わない試料 (used に含まれない label) は除いて平均を取る。
    IS が未検出のファイルは NaN (補正できないので、その値は欠損になる)。
    """
    samples = raw[~raw[LABEL_COL].map(is_std)]
    if used is not None:
        samples = samples[samples[LABEL_COL].isin(used)]
    means = samples.groupby(DILUTION_COL)[is_name].mean()
    f = raw[DILUTION_COL].map(means) / raw[is_name]
    return pd.Series(f.values, index=raw[FILE_COL].values, name="IS 係数")


def dilution_factor_of(dilution):
    if not dilution:
        return 1.0
    import re

    m = re.match(r"x(\d+(?:\.\d+)?)", str(dilution))
    return float(m.group(1)) if m else 1.0


# ---------------------------------------------------------------- 状態の判定
def measurement_status(full, ranges):
    """測定 (ファイル x 化合物) ごとの状態と理由。ranges: 化合物 -> 下限・上限 (・一点検量)。"""
    d = full[~full["is_std"]].copy()
    lo = d["compound"].map(ranges["下限"]) if "下限" in ranges else np.nan
    hi = d["compound"].map(ranges["上限"]) if "上限" in ranges else np.nan
    one = d["compound"].map(ranges["一点検量"]).fillna(True).astype(bool) if "一点検量" in ranges else True
    status, reason = [], []
    for conc, text, lod, loq, sn, l_, h_, o in zip(d["conc"], d.get("conc_text", pd.Series("", index=d.index)),
                                                   d["lod"], d["loq"], d["sn"], lo, hi, one):
        if pd.isna(conc) and text not in ("", "-----", "nan", None):
            status.append(ST_ERR), reason.append(f"数値として読めない値: {text}")
            continue
        if pd.isna(conc):
            status.append(ST_ND), reason.append("ピーク未検出 (-----)")
            continue
        notes = []
        if pd.notna(lod) and lod > 0 and conc < lod:
            status.append(ST_LOD), reason.append(f"LOD ({lod:g}) 未満")
            continue
        if pd.notna(loq) and loq > 0 and conc < loq:
            status.append(ST_LOQ), reason.append(f"LOQ ({loq:g}) 未満")
            continue
        if not (pd.notna(loq) and loq > 0):
            notes.append("LOQ 未評価 (S/N = ∞ など)")
        if o or pd.isna(h_):
            notes.append("検量範囲 未評価 (1 点検量など)")
        elif conc > h_:
            status.append(ST_OVER), reason.append(f"検量上限 ({h_:g}) 超過")
            continue
        elif pd.notna(l_) and conc < l_:
            status.append(ST_UNDER), reason.append(f"検量下限 ({l_:g}) 未満")
            continue
        status.append(ST_CHECK if notes else ST_OK), reason.append("; ".join(notes) if notes else "検量範囲内")
    d["状態"], d["理由"] = status, reason
    return d[["file", "compound", LABEL_COL, DILUTION_COL, "conc", "状態", "理由", "datetime"]]


# ---------------------------------------------------------------- 値の計算と処理履歴
@dataclass
class PipelineResult:
    values: pd.DataFrame        # 解析に使う表 (FILE / label / 希釈倍率 / 化合物...)
    provenance: pd.DataFrame    # 値ごとの処理履歴 (long)
    status: pd.DataFrame        # 採用した値の状態 (行 = 解析の行, 列 = 化合物)
    notes: list = field(default_factory=list)


def run(full, raw, compounds, settings, ranges, choices=None, merge=False, mult_dilution=False,
        mask_level=qc.MASK_NONE, mask_repl=qc.REPL_NAN, exclude_status=(ST_ERR,), used=None):
    """LabSolutions の出力から解析用の値を作り、処理履歴を返す。

    used: 解析に使う label の集合 (None = すべて)。使わない試料は IS 平均や希釈の自動選択の計算に含めず、
    処理履歴の「解析に使用」を False にする (行は残す。解析の表からは make_base で除く)。

    merge=False: ファイル (測定) ごとに 1 行 (希釈グループ別に解析する場合)
    merge=True : label ごとに 1 行 (検量範囲に基づいて 希釈なし / 希釈測定 を採用。採用できなければ 再測定候補)
    """
    notes = []
    choices = choices or {}
    if merge and not settings.dilution_known():
        # 希釈係数が既に掛かっているか分からない状態で 希釈なし と 希釈測定 を混ぜない
        notes.append("希釈の統合には「希釈係数の適用状態」の設定が必要です。未設定のため、希釈グループごとの解析にしています")
        merge = False
    meas = measurement_status(full, ranges)
    meas = meas[meas["compound"].isin(compounds)]
    samples = raw[~raw[LABEL_COL].map(is_std)]
    files = samples[[FILE_COL, LABEL_COL, DILUTION_COL]]

    # IS 係数
    if settings.is_applied():
        isf = is_factors(raw, settings.is_name, used)
        notes.append(f"IS 補正: 係数 = 希釈グループ内の {settings.is_name} 平均 / 各ファイルの {settings.is_name}")
    else:
        isf = pd.Series(1.0, index=raw[FILE_COL].values)
        if settings.app_is and settings.quant_method == QM_INTERNAL:
            notes.append("IS 補正は行いません: LabSolutions で内部標準法を使用済みのため (二重補正の防止)")
        elif settings.app_is and settings.quant_method == QM_UNSET:
            notes.append("IS 補正は行いません: 定量方式が未設定のため")
    # 希釈係数を掛けるか (換算濃度にする場合は、希釈測定の倍率も換算に含める)
    converting = bool(settings.volume_factor)
    apply_dil = settings.dilution_state == DIL_NOT_APPLIED and (merge or mult_dilution or converting)
    if (merge or mult_dilution or converting) and settings.dilution_state == DIL_UNSET:
        notes.append("希釈測定の倍率 (LabSolutions で適用済みか) が未設定のため、希釈測定の値に倍率を掛けていません。"
                     + ("希釈測定の行は換算濃度になっていません" if converting else ""))

    # 再注入 (同じ label・同じ希釈の複数ファイル)
    order = full.drop_duplicates("file").set_index("file")["datetime"]
    files = files.assign(_t=files[FILE_COL].map(order)).sort_values("_t")
    dup = files.duplicated([LABEL_COL, DILUTION_COL], keep=False)
    if dup.any():
        notes.append(f"再注入 (同じ label・同じ希釈の複数ファイル): {int(dup.sum())} ファイル。設定: {settings.reinjection}")

    m = meas.set_index(["file", "compound"])
    prov_rows = []

    def value_of(file, cpd):
        if (file, cpd) not in m.index:
            return np.nan, ST_NONE, "測定なし"
        r = m.loc[(file, cpd)]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return r["conc"], r["状態"], r["理由"]

    def convert(file, conc, dilution, label):
        f_is = float(isf.get(file, np.nan)) if settings.is_applied() else 1.0
        f_dil = dilution_factor_of(dilution) if apply_dil else 1.0
        return conc * f_is * f_dil * settings.vol_for(label), f_is, f_dil

    def meaning(dilution, f_dil):
        if not converting:
            return "バイアル中濃度"
        if dilution and f_dil == 1.0 and settings.dilution_state != DIL_APPLIED:
            return "換算濃度ではない (希釈倍率が未換算)"
        return "換算濃度"

    if not merge:
        rows = []
        first = files.drop_duplicates([LABEL_COL, DILUTION_COL], keep="first")
        use_files = files if settings.reinjection == REINJECT_MEAN else first
        for _, fr in use_files.iterrows():
            row = {FILE_COL: fr[FILE_COL], LABEL_COL: fr[LABEL_COL], DILUTION_COL: fr[DILUTION_COL]}
            for c in compounds:
                conc, stt, why = value_of(fr[FILE_COL], c)
                val, f_is, f_dil = convert(fr[FILE_COL], conc, fr[DILUTION_COL], fr[LABEL_COL])
                if stt in exclude_status:
                    val = np.nan
                row[c] = val
                prov_rows.append({LABEL_COL: fr[LABEL_COL], "行": fr[FILE_COL], "化合物": c, "採用した測定": fr[FILE_COL],
                                  "希釈": fr[DILUTION_COL] or "希釈なし", "入力値": conc, "IS 係数": f_is,
                                  "希釈係数": f_dil, "体積換算係数": settings.vol_for(fr[LABEL_COL]), "最終値": val,
                                  "最終値の意味": meaning(fr[DILUTION_COL], f_dil), "状態": stt, "理由": why,
                                  "採用しなかった測定": ""})
            rows.append(row)
        values = pd.DataFrame(rows)
        if settings.reinjection == REINJECT_MEAN and dup.any():
            # 同じ label・希釈の再注入を平均し、1 行にまとめる
            values = values.groupby([LABEL_COL, DILUTION_COL], as_index=False, sort=False).agg(
                {FILE_COL: " / ".join, **{c: "mean" for c in compounds}})
            notes.append("再注入は平均しています (状態は最初の注入のものを表示)")
        else:
            dropped = files[~files.index.isin(first.index)]
            for _, fr in dropped.iterrows():
                for c in compounds:
                    conc, stt, why = value_of(fr[FILE_COL], c)
                    prov_rows.append({LABEL_COL: fr[LABEL_COL], "行": "(不採用)", "化合物": c, "採用した測定": "",
                                      "希釈": fr[DILUTION_COL] or "希釈なし", "入力値": conc, "IS 係数": np.nan,
                                      "希釈係数": np.nan, "体積換算係数": np.nan, "最終値": np.nan, "状態": stt,
                                      "理由": "再注入: 最初の測定を採用したため不採用", "採用しなかった測定": fr[FILE_COL]})
        values = values[[FILE_COL, LABEL_COL, DILUTION_COL] + list(compounds)]
    else:
        first = files.drop_duplicates([LABEL_COL, DILUTION_COL], keep="first").set_index([LABEL_COL, DILUTION_COL])
        # 希釈の自動選択 (化合物単位) は、解析に使う試料だけで判断する
        first_used = first if used is None else first[first.index.get_level_values(0).isin(used)]
        labels = list(dict.fromkeys(files.sort_values(LABEL_COL)[LABEL_COL]))
        dil_name = next((d for d in files[DILUTION_COL].unique() if d), None)
        rows = []
        for lb in labels:
            fu = first[FILE_COL].get((lb, ""), None)
            fd = first[FILE_COL].get((lb, dil_name), None) if dil_name else None
            row = {FILE_COL: " / ".join(f for f in (fu, fd) if f), LABEL_COL: lb, DILUTION_COL: "統合"}
            for c in compounds:
                cu = value_of(fu, c) if fu else (np.nan, ST_NONE, "希釈なしの測定なし")
                cd = value_of(fd, c) if fd else (np.nan, ST_NONE, "希釈測定なし")
                pref = _preferred(c, choices, ranges, cu, cd, files, first_used, value_of, dil_name)
                cand = [(fu, "", cu), (fd, dil_name, cd)]
                if pref == "d":
                    cand = cand[::-1]
                adopted = None
                for f_, dl, (conc, stt, why) in cand:
                    if f_ and stt in ADOPTABLE:
                        adopted = (f_, dl, conc, stt, why)
                        break
                other = [(f_, dl, v) for f_, dl, v in cand if f_ and (adopted is None or f_ != adopted[0])]
                other_txt = "; ".join(f"{dl or '希釈なし'}: {v[0]:.4g} ({v[1]})" if pd.notna(v[0])
                                      else f"{dl or '希釈なし'}: {v[1]}" for f_, dl, v in other)
                if adopted is None:
                    # 採用できる測定が無い: 値は採用せず、再測定候補にする (未検出・測定なしはそのまま)
                    stts = [v[1] for f_, dl, v in cand if f_]
                    if stts and all(s in (ST_ND, ST_NONE) for s in stts):
                        stt, why = (ST_ND if ST_ND in stts else ST_NONE), "; ".join(v[2] for f_, dl, v in cand if f_)
                    elif not stts:
                        stt, why = ST_NONE, "測定なし"
                    else:
                        stt, why = ST_REMEASURE, "採用できる測定が無い (" + other_txt + ")"
                    row[c] = np.nan
                    prov_rows.append({LABEL_COL: lb, "行": lb, "化合物": c, "採用した測定": "", "希釈": "",
                                      "入力値": np.nan, "IS 係数": np.nan, "希釈係数": np.nan,
                                      "体積換算係数": settings.vol_for(lb), "最終値": np.nan, "最終値の意味": "",
                                      "状態": stt, "理由": why, "採用しなかった測定": other_txt})
                    continue
                f_, dl, conc, stt, why = adopted
                val, f_is, f_dil = convert(f_, conc, dl, lb)
                # 希釈間の一致 (両方が検量範囲内で採用可能なとき)
                if pd.notna(cu[0]) and pd.notna(cd[0]) and cu[1] in (ST_OK, ST_CHECK) and cd[1] in (ST_OK, ST_CHECK) \
                        and settings.dilution_state == DIL_NOT_APPLIED and cu[0] > 0:
                    ratio = cd[0] * dilution_factor_of(dil_name) / cu[0]
                    if abs(ratio - 1) * 100 > settings.consistency_tol:
                        stt = ST_CHECK if stt == ST_OK else stt
                        why += f"; 希釈間の不一致 (希釈測定 x 倍率 / 希釈なし = {ratio:.2f})"
                if stt in exclude_status:
                    val = np.nan
                row[c] = val
                prov_rows.append({LABEL_COL: lb, "行": lb, "化合物": c, "採用した測定": f_, "希釈": dl or "希釈なし",
                                  "入力値": conc, "IS 係数": f_is, "希釈係数": f_dil, "体積換算係数": settings.vol_for(lb),
                                  "最終値": val, "最終値の意味": meaning(dl, f_dil), "状態": stt, "理由": why,
                                  "採用しなかった測定": other_txt})
            rows.append(row)
        values = pd.DataFrame(rows, columns=[FILE_COL, LABEL_COL, DILUTION_COL] + list(compounds))

    prov = pd.DataFrame(prov_rows)
    prov.insert(1, "解析に使用", True if used is None else prov[LABEL_COL].isin(used))
    values, prov = _apply_mask(values, prov, full, compounds, mask_level, mask_repl, merge)
    keyed = prov[prov["行"] != "(不採用)"]
    status = keyed.pivot_table(index="行", columns="化合物", values="状態", aggfunc="first")
    return PipelineResult(values, prov, status, notes)


_PREF_CACHE = {}


def _preferred(c, choices, ranges, cu, cd, files, first, value_of, dil_name):
    """化合物 c で優先する測定 ("u" = 希釈なし / "d" = 希釈測定)。"""
    choice = choices.get(c, qc.CHOICE_AUTO)
    if choice == qc.CHOICE_AUTO:
        key = (id(first), c)
        if key not in _PREF_CACHE:
            _PREF_CACHE.clear() if len(_PREF_CACHE) > 5000 else None
            _PREF_CACHE[key] = _auto_compound(c, first, value_of, dil_name)
        return _PREF_CACHE[key]
    if choice == qc.SRC_NONE:
        return "u"
    if choice == qc.SRC_DIL:
        return "d"
    if choice == qc.CHOICE_AUTO_SAMPLE:
        return "d" if cu[1] == ST_OVER else "u"
    return "u"


def _auto_compound(c, first, value_of, dil_name):
    # 自動 (化合物単位): 希釈測定があるサンプルのうち 1 つでも希釈なしが上限を超えたら、化合物全体で希釈測定を優先
    if not dil_name:
        return "u"
    for (lb, dl), row in first.iterrows():
        if dl != "" or (lb, dil_name) not in first.index:
            continue
        if value_of(row[FILE_COL], c)[1] == ST_OVER:
            return "d"
    return "u"


def _apply_mask(values, prov, full, compounds, level, repl, merge):
    """LOD / LOQ 未満の値を、設定に従って置き換える (処理履歴に記録する)。"""
    prov = prov.assign(マスク="")
    if level == qc.MASK_NONE or prov.empty:
        return values, prov
    targets = {ST_LOD} if level == qc.MASK_LOD else {ST_LOD, ST_LOQ}
    col = "lod" if level == qc.MASK_LOD else "loq"
    lim = full.drop_duplicates(["file", "compound"]).set_index(["file", "compound"])[col]
    hit = prov["状態"].isin(targets) & (prov["行"] != "(不採用)")
    key = LABEL_COL if merge else FILE_COL
    for i in prov.index[hit]:
        r = prov.loc[i]
        L = lim.get((r["採用した測定"], r["化合物"]), np.nan)
        if repl == qc.REPL_NAN or pd.isna(L) or L <= 0:
            new = np.nan
        else:
            factor = r["最終値"] / r["入力値"] if r["入力値"] else 1.0
            new = L * (0.5 if repl == qc.REPL_HALF else 1.0) * factor
        prov.at[i, "マスク"] = f"{level}: {repl}"
        prov.at[i, "最終値"] = new
        mask = values[key] == (r[LABEL_COL] if merge else r["行"])
        values.loc[mask, r["化合物"]] = new
    return values, prov


# ---------------------------------------------------------------- 試料のメタデータと生物学的 n
TYPE_COL, SUBJECT_COL, TIME_COL, SERUM_COL = "試料種別", "個体ID", "時点", "血清量 (µL)"
META_COLS = [TYPE_COL, SUBJECT_COL, TIME_COL, SERUM_COL, "実験回", "前処理バッチ", "測定バッチ"]


def volume_overrides(meta, const_expr):
    """条件設定の「血清量 (µL)」が入力された試料の換算係数 {label: 係数}。"""
    out = {}
    if meta is None or SERUM_COL not in meta:
        return out
    for lb, v in meta[SERUM_COL].items():
        try:
            if str(v).strip():
                out[lb] = conversion_factor(const_expr, float(v))
        except (ValueError, ZeroDivisionError):
            pass
    return out
T_SAMPLE, T_STD = "試料", "検量線 STD"
SAMPLE_TYPES = [T_SAMPLE, "既知濃度 QC", "プール QC", "溶媒ブランク", "処理ブランク", "キャリーオーバー確認ブランク", T_STD]


def apply_metadata(base, meta, compounds):
    """メタデータを付け、試料以外 (QC・ブランク・STD) を除き、技術反復を平均して 1 個体・1 時点 1 行にする。

    同じ個体の再注入や技術反復を独立した生物学的 n として数えないため。戻り値: (表, 説明の文字列)
    """
    m = meta.reindex(base[LABEL_COL])
    m[TYPE_COL] = m[TYPE_COL].fillna(T_SAMPLE)
    m[SUBJECT_COL] = [s_ if isinstance(s_, str) and s_ else lb for lb, s_ in zip(base[LABEL_COL], m[SUBJECT_COL])]
    base = base.copy()
    for c in META_COLS:  # 以前の版の表に無い列は空欄にする
        base[c] = m[c].fillna("").values if c in m else ""
    n_other = int((base[TYPE_COL] != T_SAMPLE).sum())
    base = base[base[TYPE_COL] == T_SAMPLE]
    keys = [GROUP_COL, SUBJECT_COL, TIME_COL]
    dup = base.duplicated(keys, keep=False)
    msgs = []
    if n_other:
        msgs.append(f"試料以外 (QC・ブランクなど) の {n_other} 行を生物学的な解析から除きました")
    if dup.any():
        conflict = base[dup].groupby(keys)[COND_COL].nunique()
        if (conflict > 1).any():
            msgs.append("同じ個体ID・時点で condition が異なる行があります (最初の condition を使用): "
                        + ", ".join(map(str, conflict[conflict > 1].index.get_level_values(1)[:5])))
        num = [c for c in base.columns if c in compounds]
        agg = {c: "mean" for c in num}
        agg.update({FILE_COL: " / ".join, LABEL_COL: lambda v: " + ".join(v), COND_COL: "first", DILUTION_COL: "first"})
        agg.update({c: "first" for c in META_COLS if c not in keys})
        n_before = len(base)
        base = base.groupby(keys, as_index=False, sort=False).agg(agg)
        msgs.append(f"技術反復 (同じ個体ID・時点) {n_before - len(base)} 行を平均してまとめました")
    n_bio = base.groupby(GROUP_COL)[SUBJECT_COL].nunique().to_dict()
    msgs.append("生物学的 n (個体数): " + ", ".join(f"{g}: {n}" for g, n in n_bio.items()))
    return base, " ・ ".join(msgs)


