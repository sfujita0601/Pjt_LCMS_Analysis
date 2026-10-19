"""タブ: 選んだ代謝物のレーダーチャート (基準の condition に対する各群の評価値とばらつき)。"""
import numpy as np
import pandas as pd
import streamlit as st

import lcms_radar as lr
import lcms_stats as ls
from lcms_analysis import COND_COL, GROUP_COL
from lcms_plots import radar_plot
from ui_common import compound_select, register_output, show_chart, star_legend, stars, to_csv_bytes

_FORMULA = {
    lr.M_FC: "FC = 各群の中心値 / 基準群の中心値。ばらつきは誤差の伝搬則: SD(FC) ≈ FC·√((s₁/m₁)² + (s₀/m₀)²)",
    lr.M_LOG2FC: "log2 FC。ばらつき: √((s₁/m₁)² + (s₀/m₀)²) / ln 2 (log2 値で計算するときは √(s₁² + s₀²))",
    lr.M_PCT: "変化率 = 100 × (FC − 1)。ばらつきは FC と同じ (誤差の伝搬則) を 100 倍",
    lr.M_DIFF: "差分 = 各群の中心値 − 基準群の中心値。ばらつき: √(s₁² + s₀²) (SEM・信頼区間は √(s₁²/n₁ + s₀²/n₀))",
    lr.M_T: "Welch の t 値 = (m₁ − m₀) / √(s₁²/n₁ + s₀²/n₀)。すでにばらつきで割った値なので、ばらつきは付けません",
    lr.M_G: "Hedges の g (プールした SD で標準化し小標本補正)。ばらつきは g の標準誤差 √((n₁+n₀)/(n₁n₀) + g²/(2(n₁+n₀)))",
    lr.M_Z: "Z = (各群の中心値 − 基準群の中心値) / 基準群の SD。ばらつき: s₁ / s₀ (基準群の SD の推定誤差は含めない)",
    lr.M_VALUE: "各群の中心値。化合物ごとに桁が違う場合は「化合物ごとの尺度」をそろえてください",
}


def radar_tab(base, groups, conds, control, compounds, pal):
    st.caption("選んだ代謝物を軸にしたレーダーチャートです。基準の condition に対する各群の評価値 (Fold change・差分・t 値など) と、"
               "そのばらつきを表示します。軸が多すぎると読みにくいので、3〜20 個程度が目安です。")
    if len(conds) < 1:
        st.info("条件設定タブで condition を入力してください。")
        return
    c1, c2, c3 = st.columns([1, 2, 2])
    group = c1.selectbox("希釈グループ", groups, key="rad_group")
    metric = c2.selectbox("評価のしかた", lr.METRICS, key="rad_metric")
    ref = c3.selectbox("基準の condition", conds, index=conds.index(control) if control in conds else 0, key="rad_ref",
                       disabled=metric == lr.M_VALUE)
    sel = compound_select("rad_cpds", compounds, default=compounds[: min(8, len(compounds))], max_selections=60)
    others = [c for c in conds if c != ref] if metric != lr.M_VALUE else list(conds)
    show = st.multiselect("表示する condition (空欄 = すべて)", others, key="rad_conds", placeholder="すべて表示")
    show = [c for c in show if c in others] or others

    c1, c2, c3, c4 = st.columns(4)
    center = c1.radio("中心値", [lr.C_MEAN, lr.C_MEDIAN], key="rad_center", horizontal=True,
                      disabled=metric in (lr.M_T, lr.M_G))
    log2 = c1.checkbox("log2 変換した値で計算", value=False, key="rad_log",
                       help="平均は幾何平均に、FC は幾何平均の比に、差分・t 値・g・Z は log2 値の差になります。"
                            "濃度が右に裾を引く場合や、群の SD が平均に比例する場合に向いています")
    disp = c2.selectbox("ばらつき", lr.DISPERSIONS, key="rad_disp", disabled=metric == lr.M_T)
    style = c2.radio("ばらつきの表示", ["誤差線", "帯", "なし"], key="rad_dstyle", horizontal=True)
    include_ref = c3.checkbox("基準群自身のばらつきも描く", value=False, key="rad_ref_show",
                              disabled=metric in (lr.M_VALUE, lr.M_T, lr.M_G),
                              help="基準群を基準線の上に描き、基準群の中のばらつき (基準側の誤差だけ) を示します")
    scale = c3.selectbox("化合物ごとの尺度", [lr.S_NONE, lr.S_MAX, lr.S_Z], key="rad_scale",
                         disabled=metric != lr.M_VALUE)
    facet = c4.radio("配置", ["重ねる", "condition ごとに並べる"], key="rad_layout") != "重ねる"
    ncols = c4.number_input("1 行の図の数", 1, 6, 3, key="rad_ncols", disabled=not facet)

    with st.expander("軸の並べ方・範囲・有意差の印"):
        c1, c2, c3 = st.columns(3)
        order = c1.selectbox("軸の並べ方", ["選んだ順", "名前順", "値の大きい順 (最初の condition)",
                                          "|値| の大きい順 (最初の condition)"], key="rad_order")
        manual = c2.checkbox("半径の範囲を指定", value=False, key="rad_manual")
        rmin = c2.number_input("最小 (中心)", value=-2.0, step=0.5, key="rad_rmin", disabled=not manual)
        rmax = c2.number_input("最大 (外周)", value=2.0, step=0.5, key="rad_rmax", disabled=not manual)
        sig = c3.checkbox("基準群との検定で有意な点にアスタリスク", value=False, key="rad_sig",
                          disabled=metric == lr.M_VALUE)
        tests = [t for t in ls.TESTS]
        test = c3.selectbox("検定", tests, index=tests.index("Welch の t 検定") if "Welch の t 検定" in tests else 0,
                            key="rad_test", disabled=not sig)
        corr = c3.selectbox("多重性補正", list(ls.CORRECTIONS), key="rad_corr", disabled=not sig)
        tick_mode = c1.selectbox("円周の値の表示", ["倍率と %", "倍率", "%", "値"], key="rad_tickmode",
                                 disabled=metric not in (lr.M_FC, lr.M_LOG2FC, lr.M_PCT),
                                 help="各円の値を、基準からの倍率 (×1.5) や増減率 (+50%) で表示します")
        tick_size = c1.number_input("円周の値の文字の大きさ", 6, 30, 11, key="rad_ticksize")
        n_boot = c1.number_input("ブートストラップの回数", 200, 20000, 2000, 500, key="rad_nboot",
                                 disabled=disp != lr.D_BOOT)
    if len(sel) < 3:
        st.info("化合物を 3 つ以上選んでください (レーダーチャートは軸が 3 本以上必要です)。")
        return
    if not show:
        st.info("基準以外の condition がありません。")
        return
    sub = base[(base[GROUP_COL] == group) & (base[COND_COL] != "")]
    if metric != lr.M_VALUE and not (sub[COND_COL] == ref).any():
        st.warning(f"希釈グループ {group} に基準の condition {ref} の試料がありません。")
        return

    table = lr.radar_table(sub, sel, ref, show, metric, center, disp, log2, include_ref, int(n_boot), seed=0)
    if metric == lr.M_VALUE and scale != lr.S_NONE:
        table = lr.rescale(table, scale, sub, log2)
    drawn = ([ref] if include_ref and metric not in (lr.M_VALUE, lr.M_T, lr.M_G) else []) + list(show)
    first = table[table[COND_COL] == show[0]].set_index("化合物")["値"]
    if order == "名前順":
        axes = sorted(sel)
    elif order.startswith("値の"):
        axes = list(first.reindex(sel).sort_values(ascending=False, na_position="last").index)
    elif order.startswith("|値|"):
        axes = list(first.reindex(sel).abs().sort_values(ascending=False, na_position="last").index)
    else:
        axes = list(sel)

    star_text = None
    if sig and metric != lr.M_VALUE:
        star_text = {}
        for cond in show:
            res = ls.compare_groups(sub, axes, cond, ref, test, corr, log_for_test=True)
            for _, r in res.iterrows():
                star_text[(cond, r["化合物"])] = stars(r["q"])
                table.loc[(table[COND_COL] == cond) & (table["化合物"] == r["化合物"]), ["p", "補正後 p"]] = [r["p"], r["q"]]

    b = lr.baseline(metric)
    if log2 and metric == lr.M_VALUE:
        vt = "log2(値)"
    else:
        vt = metric.split(" (")[0] + (" (log2 値で計算)" if log2 and metric not in (lr.M_LOG2FC, lr.M_VALUE) else "")
    r_range = (rmin, rmax) if manual else _auto_range(table, b, style != "なし" and disp != lr.D_NONE)
    title = f"{vt} ・ 基準 {ref} ・ {center if metric not in (lr.M_T, lr.M_G) else '平均'}" if metric != lr.M_VALUE \
        else f"{vt} ・ {center}"
    if disp != lr.D_NONE and style != "なし" and metric != lr.M_T:
        title += f" ± {disp.split(' (')[0]}"
    fig = radar_plot(table, drawn, axes, pal, base=b, ref_name=ref, disp_style=style if disp != lr.D_NONE else "なし",
                     facet=facet, ncols=int(ncols), r_range=r_range, star_text=star_text, title=f"{title} ({group})",
                     value_title="condition", tick_mode=tick_mode, tick_size=int(tick_size),
                     tick_kind={lr.M_LOG2FC: "log2", lr.M_FC: "ratio", lr.M_PCT: "pct"}.get(metric, "value"))
    show_chart(st, fig)

    notes = [_FORMULA[metric]]
    if b is not None:
        notes.append(f"破線の円 = 基準 ({ref}) と同じ値 ({b:g})。")
    if metric != lr.M_VALUE and r_range[0] > (b if b is not None else -np.inf):
        notes.append("半径の最小値が基準線より大きいため、基準線より小さい値は中心に寄せて表示されます。")
    if disp == lr.D_BOOT:
        notes.append(f"ブートストラップ: 各群の中で復元抽出を {int(n_boot)} 回 (乱数の種 0)、2.5–97.5 パーセンタイル。"
                     "n が 5 未満では区間が狭く・不安定になりやすいので参考程度に。")
    elif disp == lr.D_CI:
        notes.append("95% 信頼区間の t 分布の自由度は Welch-Satterthwaite の近似 (多重性は補正していません)。")
    if star_text is not None:
        notes.append(f"アスタリスク: {test}・{corr} (log2 値で検定)。{star_legend()}")
    if (table["n"] < 3).any():
        notes.append("n < 3 の群・化合物があります (表の n 列)。ばらつきの推定は不安定です。")
    st.caption(" ".join(notes))
    st.caption("レーダーチャートは軸の並べ方で形 (面積) の印象が大きく変わります。面積を群間の比較の根拠にはしないでください。")

    with st.expander("表で見る"):
        st.dataframe(table.round(4), hide_index=True, height=320)
        st.download_button("レーダーチャートの値 (CSV)", to_csv_bytes(table), "radar.csv", "text/csv", key="_dl_radar")
    register_output(f"可視化/レーダーチャート ({group})", table)


def _auto_range(table, b, with_disp):
    cols = ["値", "下限", "上限"] if with_disp else ["値"]
    v = pd.to_numeric(table[cols].stack(), errors="coerce").values
    v = v[np.isfinite(v)]
    if b is not None:
        v = np.r_[v, b]
    if not len(v):
        return (0, 1)
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * 0.08 or 0.5
    lo = 0.0 if (b is None and lo >= 0) else lo - pad
    return (lo, hi + pad)
