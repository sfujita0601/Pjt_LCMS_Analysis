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
from lcms_plots import PLOT_KINDS, group_plot, resolve_plot_kind
from make_conc_table import FILE_COL, LABEL_COL
from ui_common import compound_picker, current_editor, reset_editor, stable_editor, to_csv_bytes

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
    st.plotly_chart(fig, width="stretch")

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
    st.plotly_chart(fig, width="stretch")
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
        cols[i % len(cols)].plotly_chart(fig, width="stretch", key=f"ratio_fig_{i}")
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
    c1.plotly_chart(fig, width="stretch")
    c1.caption(f"{kind}。* 補正後 p < 0.05, ** < 0.01, *** < 0.001 ({ph}"
               + (f", {ph_corr}" if ph in ls.POSTHOC_NEEDS_ADJUST else "") + (f", 対照群 {ctrl}" if ctrl else "")
               + f")。閾値 {q_thr} 以下のペアのみ表示。")
    c2.dataframe(pw.round(5), hide_index=True)
    c2.download_button("事後検定 (CSV)", to_csv_bytes(pw), f"posthoc_{cpd}.csv", "text/csv", key="_dl_ph")
