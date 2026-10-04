"""タブ: 定量限界・検出限界 / 代謝物の比 / 3 群以上の比較。"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lcms_analysis as la
import lcms_qc as qc
import lcms_stats as ls
from lcms_analysis import COND_COL, GROUP_COL, natural_key
from lcms_plots import PLOT_KINDS, color_map, group_plot, resolve_colormap, resolve_plot_kind, scale
from make_conc_table import FILE_COL, LABEL_COL
from ui_common import (
    colormap_select, compound_picker, current_editor, reset_editor, show_chart, stable_editor, to_csv_bytes,
)

LIM_COLORS = {qc.LIM_ND: "#e1e0d9", qc.LIM_LOD: "#e34948", qc.LIM_LOQ: "#eda100", qc.LIM_OK: "#2a78d6",
              qc.LIM_UNK: "#9ec5f4"}


# ---------------------------------------------------------------- 定量限界・検出限界
def limits_tab(full, pal, mask_info):
    cat = qc.limit_categories(full)
    st.caption("LabSolutions が測定ごとに S/N から算出した 検出限界 (LOD, S/N = 3) と 定量限界 (LOQ, S/N = 10) を使って、"
               "各値を分類します。ノイズが 0 で S/N が ∞ と出力された測定は限界値が 0 になるので「評価不可」とします。"
               " マスクの設定はサイドバーの「定量限界・検出限界」で行います。")
    if mask_info:
        st.info(mask_info)
    c1, c2 = st.columns([1, 3])
    samples_only = c1.checkbox("サンプルのみ (STD を除く)", value=True, key="lim_samples")
    hide_nd = c1.checkbox("全サンプル未検出の化合物を隠す", value=True, key="lim_hide")
    summ = qc.limit_summary(cat, samples_only)
    if hide_nd:
        summ = summ[summ[qc.LIM_ND] < summ[qc.LIM_ORDER].sum(axis=1)]
    c2.dataframe(summ, hide_index=True, height=300)
    c2.download_button("集計 (CSV)", to_csv_bytes(summ), "limits_summary.csv", "text/csv", key="_dl_lim")

    # 区分のヒートマップ (化合物 x 測定順)
    view = cat[~cat["is_std"]] if samples_only else cat
    inj = qc.injection_order(full)
    files = [f for f in inj["file"] if f in set(view["file"])]
    cpds = sorted(summ["化合物"], key=natural_key)
    code = {k: i for i, k in enumerate(qc.LIM_ORDER)}
    Z = view.pivot_table(index="compound", columns="file", values="区分", aggfunc="first").reindex(index=cpds, columns=files)
    C = view.pivot_table(index="compound", columns="file", values="conc", aggfunc="first").reindex(index=cpds, columns=files)
    Q = view.pivot_table(index="compound", columns="file", values="loq", aggfunc="first").reindex(index=cpds, columns=files)
    n = len(qc.LIM_ORDER)
    colorscale = []
    for k, i in code.items():
        colorscale += [[i / n, LIM_COLORS[k]], [(i + 1) / n, LIM_COLORS[k]]]
    short = [re.sub(r"^\d{8}_", "", f).replace(".lcd", "") for f in files]
    fig = go.Figure(go.Heatmap(
        z=Z.apply(lambda s: s.map(code)).values, x=short, y=cpds, zmin=0, zmax=n, colorscale=colorscale,
        xgap=1, ygap=1, customdata=np.dstack([Z.fillna("").values, C.values, Q.values]),
        colorbar=dict(tickvals=[i + 0.5 for i in range(n)], ticktext=qc.LIM_ORDER, title="区分"),
        hovertemplate="%{y}<br>%{x}<br>%{customdata[0]}<br>濃度 %{customdata[1]:.3g}<br>LOQ %{customdata[2]:.3g}"
                      "<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickfont_size=9)
    fig.update_xaxes(tickangle=-90, tickfont_size=9)
    fig.update_layout(height=max(450, 14 * len(cpds) + 200), margin=dict(l=220, t=30),
                      title="各測定の区分 (列 = 測定順)")
    show_chart(st, fig, width="stretch")

    st.subheader("化合物ごとの濃度と限界値")
    c1, c2 = st.columns([3, 1])
    cpd = c1.selectbox("化合物", cpds, key="lim_cpd")
    logy = c2.checkbox("縦軸を対数", value=True, key="lim_log")
    d = view[view["compound"] == cpd].merge(inj[["file", "order"]], on="file").sort_values("order")
    d["表示名"] = [re.sub(r"^\d{8}_", "", f).replace(".lcd", "") for f in d["file"]]
    fig = go.Figure()
    for k in qc.LIM_ORDER:
        g = d[(d["区分"] == k) & d["conc"].notna()]
        if not g.empty:
            fig.add_trace(go.Scatter(x=g["表示名"], y=g["conc"], mode="markers", name=k,
                                     marker=dict(size=10, color=LIM_COLORS[k], line=dict(width=1, color="white")),
                                     hovertemplate="%{x}<br>濃度 %{y:.3g}<extra>" + k + "</extra>"))
    for col, name, dash in [("loq", "LOQ", "solid"), ("lod", "LOD", "dot")]:
        g = d[d[col] > 0]
        fig.add_trace(go.Scatter(x=g["表示名"], y=g[col], mode="markers", name=name,
                                 marker=dict(symbol="line-ew", size=16, line=dict(width=2, color=pal["line"])),
                                 hovertemplate="%{x}<br>" + name + " %{y:.3g}<extra></extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=list(d["表示名"]), tickangle=-90, tickfont_size=9)
    fig.update_yaxes(type="log" if logy else "linear", title="濃度 (装置の算出値)")
    fig.update_layout(height=460, title=cpd, legend_title_text="")
    show_chart(st, fig, width="stretch")
    st.caption("横棒 = 測定ごとの LOQ (実線の記号) と LOD。未検出・評価不可の測定は限界値が無いため棒を描きません。")


# ---------------------------------------------------------------- 代謝物の比
RATIO_KEY = "ratios"


def ratio_definitions(compounds):
    """比の定義を編集する。[(名前, [分子], [分母])] を返す。"""
    init = pd.DataFrame({"名前": pd.Series(dtype=str), "分子": pd.Series(dtype=str), "分母": pd.Series(dtype=str)})
    st.caption("分子・分母は化合物名を「 + 」でつないで合計も指定できます (例: Valine + Leucine + Isoleucine)。"
               "分子・分母のどれかが未検出のサンプルでは比は欠損になります。比は試料ごとの正規化係数に依存しません。")
    with st.form("ratio_add", clear_on_submit=True, border=True):
        c1, c2, c3, c4 = st.columns([3, 3, 2, 1])
        num = c1.multiselect("分子 (複数選ぶと合計)", compounds, key="_ratio_num")
        den = c2.multiselect("分母 (複数選ぶと合計)", compounds, key="_ratio_den")
        name = c3.text_input("名前 (空欄なら自動)", key="_ratio_name")
        add = c4.form_submit_button("追加")
    if add and num and den:
        cur = current_editor(RATIO_KEY)
        cur = init if cur is None else cur
        row = pd.DataFrame([{"名前": name or f"{la.RATIO_SEP.join(num)} / {la.RATIO_SEP.join(den)}",
                             "分子": la.RATIO_SEP.join(num), "分母": la.RATIO_SEP.join(den)}])
        reset_editor(RATIO_KEY, pd.concat([cur, row], ignore_index=True))
        st.rerun()
    c1, c2 = st.columns([1, 3])
    if c1.button("よく使う比を追加", key="_btn_ratio_preset",
                 help="Kyn/Trp, Fischer 比, Cit/Arg, GSSG/GSH, SAM/SAH など (データにある化合物のみ)"):
        cur = current_editor(RATIO_KEY)
        cur = init if cur is None else cur
        have = set(cur["名前"].dropna())
        rows = [{"名前": n, "分子": la.RATIO_SEP.join(a), "分母": la.RATIO_SEP.join(b)}
                for n, a, b in la.RATIO_PRESETS if n not in have and all(t in compounds for t in a + b)]
        reset_editor(RATIO_KEY, pd.concat([cur, pd.DataFrame(rows)], ignore_index=True))
        st.rerun()
    edited = stable_editor(RATIO_KEY, init, num_rows="dynamic", hide_index=True, width="stretch",
                           column_config={"名前": st.column_config.TextColumn("名前"),
                                          "分子": st.column_config.TextColumn("分子", width="large"),
                                          "分母": st.column_config.TextColumn("分母", width="large")})
    ratios, errors = la.validate_ratios(edited, set(compounds))
    for e in errors:
        st.warning(e)
    return ratios


def ratio_view(base, ratios, groups, conds, pal, na_rep):
    if not ratios:
        st.info("比を追加すると、ここに表と図が表示されます。")
        return
    names = [r[0] for r in ratios]
    st.subheader("比の表")
    cols = [FILE_COL, LABEL_COL, COND_COL, GROUP_COL] + names
    tbl = base[cols]
    st.dataframe(tbl.round(5), hide_index=True, height=300)
    st.download_button("比の表 (CSV)", to_csv_bytes(tbl, na_rep), "ratios.csv", "text/csv", key="_dl_ratio")

    st.subheader("比の分布")
    c1, c2, c3 = st.columns([3, 1, 1])
    sel = c1.multiselect("比", names, default=names[:min(4, len(names))], key="ratio_sel")
    group = c2.selectbox("希釈グループ", groups, key="ratio_group")
    sub = base[base[GROUP_COL] == group].copy()
    sub[COND_COL] = sub[COND_COL].replace("", "未設定")
    cats = [c for c in conds + ["未設定"] if (sub[COND_COL] == c).any()]
    kind, pts, logy = plot_controls("ratio", False)
    cols = st.columns(min(2, max(1, len(sel))))
    for i, r in enumerate(sel):
        fig = group_plot(sub, r, COND_COL, cats, LABEL_COL, pal, kind, pts, logy)
        fig.update_layout(height=380, title=r, yaxis_title=r, margin=dict(t=40))
        show_chart(cols[i % len(cols)], fig, width="stretch", key=f"ratio_fig_{i}")
    st.caption("箱ひげ図: 箱 = 中央値と四分位、ひげ = 1.5 IQR。点 = 各サンプル。"
               "サイドバーではなくこのタブの「比を他の解析にも追加」をオンにすると、棒グラフ・ボルケーノなどでも化合物と同様に扱えます。")


# ---------------------------------------------------------------- 3 群以上の比較
def plot_controls(key, parametric):
    """検定と一緒に載せる図の種類を選ぶ。(図の種類, 点を重ねるか, 縦軸を対数か) を返す。"""
    c1, c2, c3 = st.columns([2, 1, 1])
    kind = c1.selectbox("図の種類", PLOT_KINDS, key=f"{key}_plot",
                        help="自動: 平均を比べる検定 (t 検定・ANOVA) は 平均 ± SD の棒グラフ、順位に基づく検定は箱ひげ図")
    pts = c2.checkbox("各サンプルの点を重ねる", value=True, key=f"{key}_plot_pts")
    logy = c3.checkbox("縦軸を対数", value=False, key=f"{key}_plot_log")
    return resolve_plot_kind(kind, parametric), pts, logy


def add_brackets(fig, pairs, groups, ymax, pal, logy=False):
    """有意なペアを括弧と * で示す。pairs: [(群1, 群2, 補正後 p)]"""
    if logy:
        top, step = np.log10(ymax), 0.08
        conv = lambda v: 10 ** v  # noqa: E731
    else:
        top, step = ymax, 0.08 * ymax
        conv = lambda v: v  # noqa: E731
    for level, (a, b, p) in enumerate(pairs, start=1):
        x0, x1 = groups.index(a), groups.index(b)
        y = top + step * level
        fig.add_shape(type="path", line=dict(color=pal["line"], width=1),
                      path=f"M {x0},{conv(y - step * 0.3)} L {x0},{conv(y)} L {x1},{conv(y)} L {x1},{conv(y - step * 0.3)}")
        fig.add_annotation(x=(x0 + x1) / 2, y=conv(y) if not logy else y, yref="y", showarrow=False, yshift=8,
                           text="***" if p < 0.001 else "**" if p < 0.01 else "*")
    return fig


def multigroup_tab(base, groups, conds, compounds, pal, control=None):
    if len(conds) < 3:
        st.info("condition が 3 群以上必要です (条件設定タブ)。2 群の比較はボルケーノタブで行えます。")
        return
    c1, c2, c3 = st.columns([1, 3, 1])
    group = c1.selectbox("希釈グループ", groups, key="mg_group")
    use = c2.multiselect("比較する condition (3 群以上)", conds, default=conds, key="mg_conds")
    log = c3.checkbox("ANOVA 系は log2 値で行う", value=True, key="mg_log")
    if len(use) < 3:
        st.warning("3 群以上を選んでください")
        return
    cpds = compound_picker("mg", compounds)
    c1, c2, c3 = st.columns(3)
    test = c1.selectbox("全体の検定", ls.MULTI_TESTS, key="mg_test",
                        help="ANOVA: 正規分布・等分散を仮定 / Welch: 等分散を仮定しない / Kruskal-Wallis: 順位に基づく")
    corr = c2.selectbox("化合物間の多重性補正", list(ls.CORRECTIONS), key="mg_corr")
    q_thr = c3.number_input("補正後 p の閾値", 0.0, 1.0, 0.05, 0.01, format="%.3f", key="mg_q")
    c1, c2, c3, c4 = st.columns(4)
    pairs_mode = c1.radio("事後検定の比較", [ls.PAIRS_ALL, ls.PAIRS_CONTROL], key="mg_pairs")
    ctrl = None
    if pairs_mode == ls.PAIRS_CONTROL:
        ctrl = c2.selectbox("対照群", use, index=use.index(control) if control in use else 0, key="mg_ctrl")
    ph = c3.selectbox("事後検定", ls.POSTHOCS[pairs_mode][test], key=f"mg_ph_{pairs_mode}_{test}")
    ph_corr = c4.selectbox("事後検定の補正", list(ls.CORRECTIONS), index=list(ls.CORRECTIONS).index("Holm"),
                           key="mg_ph_corr", disabled=ph not in ls.POSTHOC_NEEDS_ADJUST,
                           help="Tukey-Kramer・Games-Howell・Dunnett は検定自体が多重比較を調整しています")

    sub = base[(base[GROUP_COL] == group) & base[COND_COL].isin(use)]
    res = ls.compare_multi(sub, cpds, use, test, corr, log).sort_values("p")
    n_sig = int((res["q"] <= q_thr).sum())
    st.markdown(f"**{test}** ・ {group} ・ {len(use)} 群 ・ 補正後 p ≤ {q_thr}: **{n_sig}** 化合物 / 検定 {int(res['p'].notna().sum())}")
    st.dataframe(res.round(5), hide_index=True, height=300)
    st.download_button("全体検定の結果 (CSV)", to_csv_bytes(res), "multigroup.csv", "text/csv", key="_dl_mg")

    # 全化合物の事後検定 (一覧)
    with st.expander(f"全化合物の事後検定 ({ph}{', 対照群 ' + ctrl if ctrl else ''})"):
        allph = []
        for cpd in res.loc[res["p"].notna(), "化合物"]:
            pw = ls.posthoc(sub, cpd, use, ph, ph_corr, log, control=ctrl)
            if not pw.empty:
                allph.append(pw.assign(化合物=cpd))
        if allph:
            allph = pd.concat(allph, ignore_index=True)
            allph = allph[["化合物"] + [c for c in allph.columns if c != "化合物"]]
            st.dataframe(allph.round(5), hide_index=True, height=300)
            st.download_button("全化合物の事後検定 (CSV)", to_csv_bytes(allph), "posthoc_all.csv", "text/csv",
                               key="_dl_ph_all")

    st.subheader("化合物ごとの事後検定")
    cands = list(res.loc[res["p"].notna(), "化合物"])
    if not cands:
        return
    cpd = st.selectbox("化合物 (p の小さい順)", cands, key="mg_cpd")
    kind, pts, logy = plot_controls("mg", test in ls.PARAMETRIC)
    pw = ls.posthoc(sub, cpd, use, ph, ph_corr, log, control=ctrl)
    c1, c2 = st.columns([3, 2])
    fig = group_plot(sub, cpd, COND_COL, use, LABEL_COL, pal, kind, pts, logy)
    sig = [] if pw.empty else [(r["群1"], r["群2"], r["補正後 p"]) for _, r in pw.iterrows() if r["補正後 p"] <= q_thr]
    ymax = float(np.nanmax(sub[cpd])) if sub[cpd].notna().any() else 1.0
    add_brackets(fig, sig, use, ymax, pal, logy)
    row = res[res["化合物"] == cpd].iloc[0]
    fig.update_layout(height=480, title=f"{cpd} ・ {test}: p = {row['p']:.3g} (補正後 {row['q']:.3g})",
                      yaxis_title=cpd, margin=dict(t=50))
    show_chart(c1, fig, width="stretch")
    c1.caption(f"{kind}。* 補正後 p < 0.05, ** < 0.01, *** < 0.001 ({ph}"
               + (f", {ph_corr}" if ph in ls.POSTHOC_NEEDS_ADJUST else "") + (f", 対照群 {ctrl}" if ctrl else "")
               + f")。閾値 {q_thr} 以下のペアのみ表示。")
    c2.dataframe(pw.round(5), hide_index=True)
    c2.download_button("事後検定 (CSV)", to_csv_bytes(pw), f"posthoc_{cpd}.csv", "text/csv", key="_dl_ph")


# ---------------------------------------------------------------- 二元配置分散分析
COEF_MODE, ANOVA_MODE = "係数 (回帰係数)", "分散分析 (要因ごと)"


def _stars(p):
    return "" if pd.isna(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def twoway_factor_editor(ds, labels, cond_map):
    """label ごとに 2 つの要因の水準を入力する表。({label: (A, B)}, 要因 A の名前, 要因 B の名前)"""
    key = f"twoway::{ds}"
    c1, c2, c3, c4 = st.columns([2, 2, 1, 2])
    name_a = c1.text_input("要因 A の名前", value="要因A", key=f"tw_name_a::{ds}")
    name_b = c2.text_input("要因 B の名前", value="要因B", key=f"tw_name_b::{ds}")
    sep = c3.text_input("区切り文字", value="_", key=f"tw_sep::{ds}", max_chars=3)
    c4.write("")
    split = c4.button("condition を区切り文字で分けて入力", key=f"_btn_tw_split::{ds}",
                      help="例: condition が「HFD_Drug」なら 要因A = HFD、要因B = Drug")

    def from_condition():
        rows = []
        for lb in labels:
            parts = str(cond_map.get(lb, "")).split(sep, 1) if sep else [cond_map.get(lb, "")]
            rows.append({LABEL_COL: lb, "要因A": parts[0] if len(parts) == 2 else "",
                         "要因B": parts[1] if len(parts) == 2 else ""})
        return pd.DataFrame(rows)

    init = pd.DataFrame({LABEL_COL: labels, "要因A": [""] * len(labels), "要因B": [""] * len(labels)})
    if split:
        reset_editor(key, from_condition())
        st.rerun()
    edited = stable_editor(key, init, hide_index=True, num_rows="fixed", disabled=[LABEL_COL],
                           height=min(400, 36 * (len(labels) + 1)),
                           column_config={"要因A": st.column_config.TextColumn(name_a or "要因A"),
                                          "要因B": st.column_config.TextColumn(name_b or "要因B")})
    factors = {}
    for _, r in edited.iterrows():
        a, b = str(r.get("要因A") or "").strip(), str(r.get("要因B") or "").strip()
        if a and b:
            factors[r[LABEL_COL]] = (a, b)
    return factors, (name_a or "要因A").strip(), (name_b or "要因B").strip()


def twoway_tab(ds, base, groups, compounds, pal, labels, cond_map):
    st.caption("2 つの要因 (例: 食餌 × 薬剤) とその交互作用を、化合物ごとに 二元配置分散分析 で検定します。"
               "各サンプルの要因の水準を下の表に入力してください (condition を「HFD_Drug」のように付けていれば分割して入力できます)。")
    with st.expander("要因の設定", expanded=True):
        factors, name_a, name_b = twoway_factor_editor(ds, labels, cond_map)
    lv_a = sorted({a for a, _ in factors.values()}, key=natural_key)
    lv_b = sorted({b for _, b in factors.values()}, key=natural_key)
    if len(lv_a) < 2 or len(lv_b) < 2:
        st.info("2 つの要因それぞれに 2 つ以上の水準を入力してください。")
        return
    c1, c2, c3, c4 = st.columns(4)
    group = c1.selectbox("希釈グループ", groups, key="tw_group")
    ref_a = c2.selectbox(f"{name_a} の基準水準", lv_a, key="tw_ref_a", help="係数はこの水準との差になります")
    ref_b = c3.selectbox(f"{name_b} の基準水準", lv_b, key="tw_ref_b")
    log = c4.checkbox("log2 変換した値で行う", value=True, key="tw_log",
                      help="濃度は右に裾を引くため対数変換を推奨。係数は log2 の差 (= log2 FC) になります")
    cpds = compound_picker("tw", compounds)
    c1, c2, c3, c4 = st.columns(4)
    mode = c1.radio("ヒートマップの値", [COEF_MODE, ANOVA_MODE], key="tw_mode")
    stat = c2.selectbox("分散分析の値", ["偏η²", "F", "-log10(補正後 p)"], key="tw_stat", disabled=mode != ANOVA_MODE)
    typ = c3.radio("平方和", ["Type II", "Type III"], key="tw_typ", horizontal=True,
                   help="交互作用がある場合の主効果の定義が異なります。釣り合い型 (各セルの n が同じ) なら同じ結果")
    corr = c4.selectbox("化合物間の多重性補正", list(ls.CORRECTIONS), key="tw_corr")
    sort = c1.selectbox("化合物の並び", ["入力順", "交互作用の p が小さい順", f"{name_a} の p が小さい順",
                                        f"{name_b} の p が小さい順"], key="tw_sort")
    with c4:
        cmap = colormap_select("tw_cmap")

    sub = base[(base[GROUP_COL] == group) & base[LABEL_COL].isin(factors)].copy()
    sub["A"] = sub[LABEL_COL].map(lambda lb: factors[lb][0])
    sub["B"] = sub[LABEL_COL].map(lambda lb: factors[lb][1])
    cells = sub.groupby(["A", "B"]).size().unstack(fill_value=0).reindex(index=lv_a, columns=lv_b, fill_value=0)
    with st.expander("各組み合わせのサンプル数"):
        st.dataframe(cells.rename_axis(index=name_a, columns=name_b))
    anova, coefs, skipped = ls.twoway_anova(sub, cpds, name_a, name_b, log, 3 if typ == "Type III" else 2, ref_a, ref_b)
    if anova.empty:
        st.warning("解析できる化合物がありません" + (f" (例: {skipped[0][0]}: {skipped[0][1]})" if skipped else ""))
        return
    method = ls.CORRECTIONS[corr]
    anova["補正後 p"] = anova.groupby("要因")["p"].transform(lambda p: ls.adjust_p(p, method).values)
    coefs["補正後 p"] = coefs.groupby("係数")["p"].transform(lambda p: ls.adjust_p(p, method).values)

    order = list(dict.fromkeys(anova["化合物"]))
    term_ab, term_a, term_b = f"{name_a}×{name_b}", name_a, name_b
    key_term = {"交互作用の p が小さい順": term_ab, f"{name_a} の p が小さい順": term_a,
                f"{name_b} の p が小さい順": term_b}.get(sort)
    if key_term:
        order = list(anova[anova["要因"] == key_term].sort_values("p")["化合物"])

    if mode == COEF_MODE:
        M = coefs.pivot(index="係数", columns="化合物", values="値")
        P = coefs.pivot(index="係数", columns="化合物", values="補正後 p")
        rows = list(dict.fromkeys(coefs["係数"]))
        unit = "log2 の差" if log else "差"
        title = f"回帰係数 ({unit}、基準: {name_a} = {ref_a}, {name_b} = {ref_b})"
        zmid, finite = 0, M.values[np.isfinite(M.values)]
        m = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
        zrange, cs = (-m, m), resolve_colormap(cmap, scale(pal["div"]))
    else:
        val = {"偏η²": "偏η²", "F": "F"}.get(stat)
        A = anova.assign(v=anova[val] if val else -np.log10(anova["補正後 p"]))
        M = A.pivot(index="要因", columns="化合物", values="v")
        P = anova.pivot(index="要因", columns="化合物", values="補正後 p")
        rows = [term_a, term_b, term_ab]
        title, zmid, zrange = stat, None, None
        cs = resolve_colormap(cmap, scale(pal["seq"]))
    M, P = M.reindex(index=rows, columns=order), P.reindex(index=rows, columns=order)
    stars = P.map(_stars)
    fig = go.Figure(go.Heatmap(
        z=M.values, x=M.columns, y=M.index, text=stars.values, texttemplate="%{text}", textfont=dict(size=14),
        colorscale=cs, zmid=zmid, zmin=zrange[0] if zrange else None, zmax=zrange[1] if zrange else None,
        xgap=1, ygap=1, customdata=P.values,
        colorbar=dict(title=("係数<br>(log2)" if log else "係数") if mode == COEF_MODE else stat),
        hovertemplate="%{x}<br>%{y}<br>値 %{z:.3g}<br>補正後 p %{customdata:.3g}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-60, tickfont_size=10)
    fig.update_layout(height=max(320, 60 * len(rows) + 220), title=title, margin=dict(l=160, b=160, t=60))
    show_chart(st, fig)
    p_label = "回帰係数の t 検定" if mode == COEF_MODE else f"分散分析 ({typ})"
    st.caption(f"* 補正後 p < 0.05, ** < 0.01, *** < 0.001 ({p_label}, {corr}, 行ごとに化合物間で補正)。"
               + ("主効果の係数は「もう一方の要因が基準水準のときの差」、交互作用の係数は「差の差」です。"
                  if mode == COEF_MODE else ""))
    if skipped:
        st.caption("解析できなかった化合物: " + ", ".join(f"{c} ({r})" for c, r in skipped[:10])
                   + (" ..." if len(skipped) > 10 else ""))

    t1, t2 = st.tabs(["分散分析表", "回帰係数"])
    with t1:
        st.dataframe(anova.round(5), hide_index=True, height=300)
        st.download_button("分散分析表 (CSV)", to_csv_bytes(anova), "twoway_anova.csv", "text/csv", key="_dl_tw_a")
    with t2:
        st.dataframe(coefs.round(5), hide_index=True, height=300)
        st.download_button("回帰係数 (CSV)", to_csv_bytes(coefs), "twoway_coefficients.csv", "text/csv", key="_dl_tw_c")

    st.subheader("交互作用の図")
    cpd = st.selectbox("化合物", order, key="tw_cpd")
    d = sub.dropna(subset=[cpd])
    summ = d.groupby(["A", "B"])[cpd].agg(["mean", "std", "count"]).reset_index()
    summ["se"] = summ["std"] / np.sqrt(summ["count"])
    cmap_b = color_map(lv_b, pal)
    fig = go.Figure()
    for b in lv_b:
        s = summ[summ["B"] == b].set_index("A").reindex(lv_a)
        fig.add_trace(go.Scatter(x=lv_a, y=s["mean"], mode="lines+markers", name=b, line=dict(color=cmap_b[b], width=2),
                                 marker=dict(size=9), error_y=dict(type="data", array=s["se"], thickness=1.5, width=6),
                                 hovertemplate=f"{name_b} = {b}<br>%{{x}}<br>平均 %{{y:.4g}}<extra></extra>"))
        pts = d[d["B"] == b]
        fig.add_trace(go.Scatter(x=pts["A"], y=pts[cpd], mode="markers", showlegend=False, text=pts[LABEL_COL],
                                 marker=dict(size=6, color=cmap_b[b], opacity=0.45),
                                 hovertemplate="%{text}<br>%{y:.4g}<extra></extra>"))
    rows_c = anova[anova["化合物"] == cpd].set_index("要因")
    sub_title = " ・ ".join(f"{t}: p = {rows_c.loc[t, 'p']:.3g}" for t in rows_c.index)
    fig.update_xaxes(title=name_a, categoryorder="array", categoryarray=lv_a)
    fig.update_yaxes(title=cpd)
    fig.update_layout(height=440, title=f"{cpd}<br><sup>{sub_title}</sup>", legend_title_text=name_b,
                      margin=dict(t=80))
    show_chart(st, fig)
    st.caption("点 = 平均 ± 標準誤差 (薄い点は各サンプル)。線が平行でなければ交互作用があることを示します。"
               "図は変換前の値、検定は" + ("log2 変換した値" if log else "そのままの値") + "です。")
