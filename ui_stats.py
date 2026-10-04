"""統計タブ: ボルケーノプロット / 相関ヒートマップ・ネットワーク / KEGG 経路図への重ね描き。"""
import io

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import lcms_kegg as lk
import lcms_stats as ls
from lcms_analysis import COND_COL, GROUP_COL, log10_safe, natural_key
from lcms_plots import color_map, resolve_colormap, scale
from make_conc_table import LABEL_COL
from lcms_plots import group_plot
from ui_common import colormap_select, compound_picker, reset_editor, show_chart, stable_editor, symmetric_limit, to_csv_bytes

UP, DOWN, NS = "増加", "減少", "有意差なし"


# ---------------------------------------------------------------- ボルケーノ
def volcano_tab(base, groups, conds, control, compounds, pal):
    """比較の結果 dict (KEGG・相関タブでも使う) を返す。比較できないときは None。"""
    others = [c for c in conds if c != control]
    if control is None or not others:
        st.info("条件設定タブで対照群と、比較する condition を 1 つ以上設定してください。")
        return None
    c1, c2, c3, c4 = st.columns(4)
    group = c1.selectbox("希釈グループ", groups, key="vol_group")
    treat = c2.selectbox(f"比較群 (vs 対照群 {control})", others, key="vol_treat")
    test = c3.selectbox("検定", ls.TESTS, key="vol_test")
    corr = c4.selectbox("多重性補正", list(ls.CORRECTIONS), key="vol_corr")
    c1, c2, c3, c4, c5 = st.columns(5)
    log_test = c1.checkbox("t 検定は log2 値で行う", value=True, key="vol_log", disabled="t 検定" not in test,
                           help="濃度は右に裾を引く分布になりやすいため、対数変換してから t 検定するのが一般的です")
    fc_thr = c2.number_input("|log2FC| の閾値", 0.0, 10.0, 1.0, 0.25, key="vol_fc")
    q_thr = c3.number_input("補正後 p の閾値", 0.0, 1.0, 0.05, 0.01, format="%.3f", key="vol_q")
    yaxis = c4.radio("縦軸", ["-log10(p)", "-log10(補正後 p)"], key="vol_y")
    n_lab = c5.number_input("ラベルを付ける上位数", 0, 100, 15, key="vol_nlab")
    compounds = compound_picker("vol", compounds)

    sub = base[base[GROUP_COL] == group]
    res = ls.compare_groups(sub, compounds, treat, control, test, corr, log_test)
    res["判定"] = np.select([(res["q"] <= q_thr) & (res["log2FC"] >= fc_thr),
                             (res["q"] <= q_thr) & (res["log2FC"] <= -fc_thr)], [UP, DOWN], NS)
    y = -np.log10(res["p"] if yaxis == "-log10(p)" else res["q"])
    res["-log10"] = y
    plot = res[np.isfinite(res["log2FC"]) & np.isfinite(y)]

    colors = {UP: pal["div"][-1], DOWN: pal["div"][0], NS: "#b5b3ac"}
    fig = go.Figure()
    for k in [NS, DOWN, UP]:
        g = plot[plot["判定"] == k]
        fig.add_trace(go.Scatter(
            x=g["log2FC"], y=g["-log10"], mode="markers", name=f"{k} ({len(g)})",
            marker=dict(size=9 if k != NS else 7, color=colors[k], line=dict(width=1, color="rgba(255,255,255,0.8)")),
            text=g["化合物"], customdata=np.stack([g["FC"], g["p"], g["q"]], axis=-1),
            hovertemplate="%{text}<br>log2FC %{x:.3f} (FC %{customdata[0]:.3g})<br>p %{customdata[1]:.3g}<br>"
                          "補正後 p %{customdata[2]:.3g}<extra></extra>",
        ))
    lab = plot[plot["判定"] != NS].nsmallest(int(n_lab), "q")
    for _, r in lab.iterrows():
        fig.add_annotation(x=r["log2FC"], y=r["-log10"], text=r["化合物"], showarrow=False, yshift=11, font_size=10)
    for xv in (-fc_thr, fc_thr):
        fig.add_vline(x=xv, line=dict(color="#898781", width=1, dash="dot"))
    if yaxis != "-log10(p)" or corr == "補正なし":
        fig.add_hline(y=-np.log10(q_thr), line=dict(color="#898781", width=1, dash="dot"))
    fig.update_layout(height=560, xaxis_title=f"log2 FC ({treat} / {control})", yaxis_title=yaxis,
                      title=f"{group} ・ {treat} vs {control} ・ {test} ・ {corr}", legend_title_text="")
    show_chart(st, fig, width="stretch")
    st.caption(f"FC = {treat} の平均 / {control} の平均。点線 = 閾値 (|log2FC| ≥ {fc_thr}, 補正後 p ≤ {q_thr})。"
               " どちらかの群で検出が 2 未満の化合物は検定せず除外しています。")
    out = res.drop(columns=["-log10"]).sort_values("p")
    st.dataframe(out.round(5), hide_index=True, height=320)
    st.download_button("検定結果 (CSV)", to_csv_bytes(out), f"volcano_{treat}_vs_{control}.csv", "text/csv", key="_dl_vol")

    st.subheader("化合物ごとの比較")
    from ui_extra import add_brackets, plot_controls  # 循環 import を避けるためここで読み込む

    cands = list(out.loc[out["p"].notna(), "化合物"])
    if cands:
        cpd = st.selectbox("化合物 (p の小さい順)", cands, key="vol_cpd")
        kind, pts, logy = plot_controls("vol", "t 検定" in test)
        two = sub[sub[COND_COL].isin([control, treat])]
        fig = group_plot(two, cpd, COND_COL, [control, treat], LABEL_COL, pal, kind, pts, logy)
        r = out[out["化合物"] == cpd].iloc[0]
        if r["q"] <= q_thr:
            add_brackets(fig, [(control, treat, r["q"])], [control, treat], float(np.nanmax(two[cpd])), pal, logy)
        fig.update_layout(height=440, yaxis_title=cpd, margin=dict(t=50),
                          title=f"{cpd} ・ {test}: p = {r['p']:.3g} (補正後 {r['q']:.3g}) ・ FC = {r['FC']:.3g}")
        c1, c2 = st.columns([2, 1])
        show_chart(c1, fig, width="stretch")
        c2.caption(f"{kind}。* 補正後 p < 0.05, ** < 0.01, *** < 0.001 ({corr})。")
    return dict(group=group, treat=treat, control=control, res=res, q_thr=q_thr, test=test, corr=corr)


# ---------------------------------------------------------------- 散布図
UNSET_COND = "未設定"


def make_square(fig, xs, ys, logx, logy, size):
    """描画領域を正方形にする (x 軸と y 軸の長さをそろえる)。

    軸の範囲を明示し、縦横の目盛りの比をデータ範囲の比に合わせて固定する (scaleanchor)。
    こうすると画面の幅によらず、描画領域は高さ size に合わせた正方形になる。
    """
    def span(v, log):
        v = np.asarray(v, dtype=float)
        v = np.log10(v[v > 0]) if log else v[np.isfinite(v)]
        lo, hi = float(v.min()), float(v.max())
        pad = (hi - lo) * 0.08 or (abs(hi) * 0.1 or 1.0)
        return lo - pad, hi + pad

    (x0, x1), (y0, y1) = span(xs, logx), span(ys, logy)
    fig.update_xaxes(range=[x0, x1], constrain="domain")
    fig.update_yaxes(range=[y0, y1], scaleanchor="x", scaleratio=(x1 - x0) / (y1 - y0), constrain="domain")
    # 凡例は下に置き、横幅を正方形の描画領域に使えるようにする
    fig.update_layout(height=size + 170, legend=dict(orientation="h", y=-0.18, x=0, yanchor="top"),
                      margin=dict(t=70, b=120))


def _fit_line(x, y, logx, logy):
    """表示している軸のスケールで直線を当てはめ、描画用の (xs, ys) を返す。"""
    tx = np.log10(x) if logx else x
    ty = np.log10(y) if logy else y
    k, b = np.polyfit(tx, ty, 1)
    xs = np.linspace(tx.min(), tx.max(), 50)
    ys = k * xs + b
    return (10 ** xs if logx else xs), (10 ** ys if logy else ys)


def _corr_text(x, y, method):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return f"n={len(x)}"
    r, p = (stats.pearsonr if method == "Pearson" else stats.spearmanr)(x, y)
    return f"n={len(x)}, r={r:.2f}, p={p:.2g}"


def scatter_tab(base, groups, conds, compounds, pal):
    if len(compounds) < 2:
        st.info("化合物が 2 つ以上必要です")
        return
    c1, c2, c3, c4 = st.columns(4)
    group = c1.selectbox("希釈グループ", groups, key="sc_group")
    x = c2.selectbox("横軸の化合物", compounds, index=0, key="sc_x")
    y = c3.selectbox("縦軸の化合物", compounds, index=1, key="sc_y")
    color_by = c4.radio("色分け", ["condition", "なし"], key="sc_color", horizontal=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    logx = c1.checkbox("横軸を対数", value=False, key="sc_logx")
    logy = c2.checkbox("縦軸を対数", value=False, key="sc_logy")
    fit = c3.selectbox("回帰直線", ["全体", "condition ごと", "なし"], key="sc_fit")
    method = c4.selectbox("相関係数", ["Spearman", "Pearson"], key="sc_method",
                          help="対数軸のときは Pearson も対数値で計算します")
    labels_on = c5.checkbox("ラベルを表示", value=True, key="sc_lab")
    c1, c2 = st.columns([1, 2])
    square = c1.checkbox("x 軸と y 軸を同じ長さにする (正方形)", value=True, key="sc_square")
    size = c2.slider("図の大きさ (px)", 300, 1000, 560, 20, key="sc_size", disabled=not square)

    sub = base[base[GROUP_COL] == group].copy()
    sub[COND_COL] = sub[COND_COL].replace("", UNSET_COND)
    d = sub[[LABEL_COL, COND_COL, x, y]].dropna()
    if logx:
        d = d[d[x] > 0]
    if logy:
        d = d[d[y] > 0]
    if len(d) < 2:
        st.warning("両方の化合物が検出されたサンプルが 2 つ未満です")
        return
    cats = [c for c in conds + [UNSET_COND] if (d[COND_COL] == c).any()] if color_by == "condition" else ["全サンプル"]
    cmap = color_map(cats, pal)
    tx = lambda v, lg: np.log10(v) if (lg and method == "Pearson") else v  # noqa: E731
    fig = go.Figure()
    for k in cats:
        g = d if color_by == "なし" else d[d[COND_COL] == k]
        name = k if fit != "condition ごと" else f"{k} ({_corr_text(tx(g[x], logx), tx(g[y], logy), method)})"
        fig.add_trace(go.Scatter(
            x=g[x], y=g[y], mode="markers+text" if labels_on else "markers", name=name, legendgroup=k,
            text=g[LABEL_COL], textposition="top center", textfont=dict(size=10),
            marker=dict(size=10, color=cmap[k], line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
            customdata=g[COND_COL],
            hovertemplate="%{text} (%{customdata})<br>" + x + ": %{x:.4g}<br>" + y + ": %{y:.4g}<extra></extra>",
        ))
        if fit == "condition ごと" and len(g) >= 3:
            xs, ys = _fit_line(g[x].values, g[y].values, logx, logy)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=cmap[k], width=2), legendgroup=k,
                                     showlegend=False, hoverinfo="skip"))
    title = f"{group} ・ {x} vs {y}"
    if fit == "全体" and len(d) >= 3:
        xs, ys = _fit_line(d[x].values, d[y].values, logx, logy)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="回帰直線 (全体)",
                                 line=dict(color=pal["line"], width=2, dash="dash"), hoverinfo="skip"))
        title += f" ・ {method}: {_corr_text(tx(d[x], logx), tx(d[y], logy), method)}"
    fig.update_xaxes(title=x, type="log" if logx else "linear")
    fig.update_yaxes(title=y, type="log" if logy else "linear")
    fig.update_layout(height=560, title=title, legend_title_text="condition" if color_by == "condition" else "")
    if square:
        make_square(fig, d[x], d[y], logx, logy, size)
    show_chart(st, fig, width="stretch")
    st.caption("両方の化合物が検出されたサンプルだけを描いています。回帰直線は表示している軸のスケール (対数軸なら対数値) で当てはめています。")

    st.subheader("散布図行列")
    dims = st.multiselect("化合物 (2〜6 個)", compounds, default=[x, y], max_selections=6, key="sc_dims")
    if len(dims) >= 2:
        n = len(dims)
        m = sub[[LABEL_COL, COND_COL] + dims]
        fig = make_subplots(rows=n, cols=n, horizontal_spacing=0.02, vertical_spacing=0.02)
        for i, yi in enumerate(dims, start=1):
            for j, xj in enumerate(dims, start=1):
                if i == j:
                    for k in cats:
                        g = m if color_by == "なし" else m[m[COND_COL] == k]
                        fig.add_trace(go.Histogram(x=g[xj], marker_color=cmap[k], opacity=0.7, showlegend=False,
                                                   legendgroup=k, nbinsx=12), row=i, col=j)
                    continue
                for k in cats:
                    g = m if color_by == "なし" else m[m[COND_COL] == k]
                    g = g.dropna(subset=[xj, yi])
                    fig.add_trace(go.Scatter(
                        x=g[xj], y=g[yi], mode="markers", name=k, legendgroup=k, showlegend=(i, j) == (1, 2),
                        marker=dict(size=6, color=cmap[k]), text=g[LABEL_COL],
                        hovertemplate="%{text}<br>" + xj + ": %{x:.3g}<br>" + yi + ": %{y:.3g}<extra></extra>",
                    ), row=i, col=j)
                if logx or logy:
                    fig.update_xaxes(type="log" if logx else "linear", row=i, col=j)
                    fig.update_yaxes(type="log" if logy else "linear", row=i, col=j)
            fig.update_yaxes(title_text=yi, title_font_size=10, row=i, col=1)
        for j, xj in enumerate(dims, start=1):
            fig.update_xaxes(title_text=xj, title_font_size=10, row=n, col=j)
        fig.update_xaxes(tickfont_size=9, nticks=4)
        fig.update_yaxes(tickfont_size=9, nticks=4)
        if square:  # 各パネルを正方形に: 図全体の幅と高さをそろえて、幅は固定で表示する
            side = max(160 * n + 120, size)
            fig.update_layout(height=side, width=side, barmode="overlay", margin=dict(t=30, l=70, r=30, b=60))
            show_chart(st, fig, width="content")
        else:
            fig.update_layout(height=160 * n + 120, barmode="overlay", margin=dict(t=30))
            show_chart(st, fig, width="stretch")
        st.caption("対角 = 各化合物のヒストグラム。対数軸の設定は上の散布図と共通です。")


# ---------------------------------------------------------------- 相関
@st.cache_data(show_spinner="相関を計算中...")
def _correlation(X, method, min_pairs):
    return ls.correlation(X, method, min_pairs)


@st.cache_data(show_spinner=False)
def _layout(edges, nodes, seed):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    return nx.spring_layout(G, seed=seed, weight="weight", k=3.0 / np.sqrt(max(1, len(nodes))), iterations=200)


def correlation_tab(base, groups, conds, compounds, pal, vres):
    c1, c2, c3, c4 = st.columns(4)
    group = c1.selectbox("希釈グループ", groups, key="cor_group")
    use_conds = c2.multiselect("使う condition (空欄 = 全サンプル)", conds, key="cor_conds")
    method = c3.selectbox("相関係数", list(ls.CORR_METHODS), key="cor_method")
    min_pairs = c4.number_input("最小サンプル数 (ペアごと)", 3, 100, 6, key="cor_min",
                                help="両方の化合物が検出されたサンプルがこの数未満のペアは計算しません")
    log = c1.checkbox("log10 変換してから計算", value=True, key="cor_log",
                      help="Pearson は外れ値に敏感なので対数変換を推奨。Spearman / Kendall は順位を使うので結果は変わりません")

    compounds = compound_picker("cor", compounds)
    sub = base[base[GROUP_COL] == group]
    if use_conds:
        sub = sub[sub[COND_COL].isin(use_conds)]
    X = sub[compounds].astype(float)
    X = X.loc[:, X.notna().sum() >= min_pairs]
    if X.shape[1] < 3:
        st.warning("相関を計算できる化合物が足りません")
        return
    if log:
        X = log10_safe(X)
    R, P, N = _correlation(X, ls.CORR_METHODS[method], int(min_pairs))
    st.caption(f"{len(sub)} サンプル x {X.shape[1]} 化合物 ・ {method} の相関係数")

    t_hm, t_net = st.tabs(["相関ヒートマップ", "相関ネットワーク"])
    with t_hm:
        c1, c2 = st.columns([1, 3])
        cluster = c1.checkbox("階層的クラスタリングで並べ替え", value=True, key="cor_cluster")
        with c1:
            cmap = colormap_select("cor_cmap")
        order = list(R.index)
        if cluster:
            D = np.array((1 - R.fillna(0)).clip(lower=0).values, copy=True)
            np.fill_diagonal(D, 0)
            order = list(R.index[leaves_list(linkage(squareform(D, checks=False), "average"))])
        Ro, Po, No = R.loc[order, order], P.loc[order, order], N.loc[order, order]
        fig = go.Figure(go.Heatmap(
            z=Ro.values, x=order, y=order, zmin=-1, zmax=1, zmid=0, xgap=1, ygap=1,
            colorscale=resolve_colormap(cmap, scale(pal["div"])), colorbar=dict(title="r"),
            customdata=np.dstack([Po.values, No.values]),
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<br>p = %{customdata[0]:.3g}<br>n = %{customdata[1]}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed", tickfont_size=9)
        fig.update_xaxes(tickangle=-90, tickfont_size=9)
        size = max(600, 14 * len(order) + 220)
        fig.update_layout(height=size, margin=dict(l=200, b=200, t=30))
        show_chart(st, fig, width="stretch")
        st.download_button("相関係数の行列 (CSV)", to_csv_bytes(R, index=True), "correlation.csv", "text/csv", key="_dl_cor")

    with t_net:
        c1, c2, c3, c4 = st.columns(4)
        r_min = c1.slider("|r| の下限", 0.0, 1.0, 0.7, 0.05, key="net_r")
        q_max = c2.number_input("補正後 p の上限", 0.0, 1.0, 0.05, 0.01, format="%.3f", key="net_q")
        corr = c3.selectbox("多重性補正", [k for k in ls.CORRECTIONS], key="net_corr")
        color_by_fc = c4.checkbox("ノードを log2FC で色付け", value=vres is not None and vres["group"] == group,
                                  disabled=vres is None, key="net_fc",
                                  help="ボルケーノタブの比較 (比較群 / 対照群) の log2FC で色を付けます")
        edges = ls.correlation_edges(R, P, N, r_min, q_max, ls.CORRECTIONS[corr] or "fdr_bh")
        if corr == "補正なし":
            edges = edges.assign(q=edges["p"])
            edges = edges[edges["q"] <= q_max]
        if edges.empty:
            st.info("条件を満たす相関がありません。閾値を緩めてください。")
            return
        nodes = sorted(set(edges["化合物1"]) | set(edges["化合物2"]), key=natural_key)
        pos = _layout(tuple(zip(edges["化合物1"], edges["化合物2"], edges["r"].abs())), tuple(nodes), 42)
        deg = pd.concat([edges["化合物1"], edges["化合物2"]]).value_counts()
        fig = go.Figure()
        for sign, color in [(1, pal["div"][-1]), (-1, pal["div"][0])]:
            e = edges[np.sign(edges["r"]) == sign]
            xs, ys = [], []
            for a, b in zip(e["化合物1"], e["化合物2"]):
                xs += [pos[a][0], pos[b][0], None]
                ys += [pos[a][1], pos[b][1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.2), opacity=0.6,
                                     hoverinfo="skip", name="正の相関" if sign > 0 else "負の相関"))
        marker = dict(size=[8 + 2.5 * deg[n] for n in nodes], line=dict(width=1, color="rgba(255,255,255,0.9)"))
        hover = [f"{n}<br>次数 {deg[n]}" for n in nodes]
        if color_by_fc and vres is not None:
            fc = vres["res"].set_index("化合物")["log2FC"].reindex(nodes)
            m = symmetric_limit(fc.values)
            marker.update(color=fc.values, colorscale=scale(pal["div"]), cmin=-m, cmax=m, cmid=0,
                          colorbar=dict(title=f"log2FC<br>{vres['treat']}/{vres['control']}"))
            hover = [h + f"<br>log2FC {v:.3f}" for h, v in zip(hover, fc.values)]
        else:
            marker["color"] = pal["series"][0]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=nodes,
            textposition="top center", textfont=dict(size=10), marker=marker, hovertext=hover, hoverinfo="text",
            showlegend=False))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        fig.update_layout(height=700, legend=dict(orientation="h", y=1.02), margin=dict(t=40))
        show_chart(st, fig, width="stretch")
        st.caption(f"エッジ {len(edges)} 本 (赤 = 正の相関, 青 = 負の相関)・ノードの大きさ = 次数。"
                   "配置は spring layout (Fruchterman-Reingold, seed 固定)。相関は因果関係を意味しません (Camacho et al., 2005)。")
        st.dataframe(edges.round(5), hide_index=True, height=300)
        st.download_button("エッジ一覧 (CSV)", to_csv_bytes(edges), "correlation_edges.csv", "text/csv", key="_dl_edges")


# ---------------------------------------------------------------- KEGG
@st.cache_data(show_spinner="KEGG から経路を検索中...", ttl=86400)
def _pathways(ids):
    return lk.pathways_for(ids), lk.pathway_names()


@st.cache_data(show_spinner="KEGG から経路図を取得中...", ttl=86400)
def _pathway_layout(pid, org):
    return lk.pathway_layout(pid, org)


def kegg_mapping_editor(all_compounds):
    """化合物 -> KEGG ID の対応表 (編集可)。"""
    base = lk.load_mapping()
    init = pd.DataFrame({"化合物": all_compounds, "kegg_id": [base.get(c, "") for c in all_compounds]})
    with st.expander(f"化合物と KEGG Compound ID の対応表 (登録 {int((init['kegg_id'] != '').sum())} / {len(init)})"):
        st.caption("resources/kegg_compounds.csv を初期値にしています (KEGG で名称を照合済み)。"
                   "経路図上の ID と異なる場合 (立体異性体など) はここで修正してください。")
        edited = stable_editor("kegg_map", init, hide_index=True, num_rows="fixed", height=300, disabled=["化合物"])
        c1, c2 = st.columns(2)
        c1.download_button("対応表 (CSV)", to_csv_bytes(edited.rename(columns={"化合物": "compound"})),
                           "kegg_compounds.csv", "text/csv", key="_dl_kegg_map")
        if c2.button("初期値に戻す", key="_btn_kegg_map_reset"):
            reset_editor("kegg_map", init)
            st.rerun()
    return {c: str(k).strip() for c, k in zip(edited["化合物"], edited["kegg_id"]) if str(k).strip()}


def kegg_tab(all_compounds, vres, pal):
    st.caption("KEGG PATHWAY の経路図に、対照群に対する濃度の増減 (log2FC) を色で重ねます。"
               "比較の設定 (希釈グループ・比較群・検定・多重性補正) はボルケーノタブのものを使います。"
               " 出典: KEGG (Kanehisa & Goto, 2000; Kanehisa et al., 2023), https://www.kegg.jp/")
    mapping = kegg_mapping_editor(all_compounds)
    if vres is None:
        st.info("ボルケーノタブで比較 (対照群と比較群) を設定してください。")
        return
    res = vres["res"][np.isfinite(vres["res"]["log2FC"])].copy()
    res["kegg_id"] = res["化合物"].map(mapping)
    res = res.dropna(subset=["kegg_id"])
    if res.empty:
        st.warning("KEGG ID が付いた化合物がありません")
        return

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    org = c1.selectbox("経路図", ["map", "hsa", "mmu", "rno"], key="kegg_org",
                       format_func=lambda o: {"map": "参照経路 (map)", "hsa": "ヒト (hsa)", "mmu": "マウス (mmu)",
                                              "rno": "ラット (rno)"}[o],
                       help="生物種を選ぶと、その生物に存在する酵素が緑色で示された図になります")
    metab_only = c2.checkbox("代謝経路のみ (map00xxx)", value=True, key="kegg_metab")
    with c3:
        cmap = colormap_select("kegg_cmap")
    limit = c4.number_input("色の上限 |log2FC| (0 = 自動)", 0.0, 20.0, 0.0, 0.5, key="kegg_lim")
    size = c5.slider("丸の大きさ", 8, 30, 16, key="kegg_size")
    try:
        pw, names = _pathways(tuple(sorted(set(res["kegg_id"]))))
    except Exception as e:
        st.error(f"KEGG に接続できません: {e}")
        return
    sig = set(res.loc[res["q"] <= vres["q_thr"], "kegg_id"])
    table = pd.DataFrame([{"経路": p, "名称": names.get(p, ""), "測定化合物数": len(set(v)),
                           "有意な化合物数": len(set(v) & sig)} for p, v in pw.items()])
    if metab_only:
        table = table[table["経路"].str.match(r"map00\d{3}")]
    if table.empty:
        st.warning("該当する経路がありません")
        return
    table = table.sort_values(["測定化合物数", "有意な化合物数"], ascending=False).reset_index(drop=True)
    labels = {r["経路"]: f"{r['経路']}  {r['名称']}  (測定 {r['測定化合物数']} / 有意 {r['有意な化合物数']})"
              for _, r in table.iterrows()}
    pid = st.selectbox("経路 (測定化合物の多い順)", table["経路"], format_func=labels.get, key="kegg_pid")
    with st.expander("経路の一覧"):
        st.dataframe(table, hide_index=True, height=300)

    try:
        png, nodes, title = _pathway_layout(pid, org)
    except FileNotFoundError:
        st.error(f"{org}{pid[-5:]} の経路図は KEGG にありません (この生物種には存在しない経路の可能性があります)。")
        return
    except Exception as e:
        st.error(f"KEGG から経路図を取得できません: {e}")
        return
    img = Image.open(io.BytesIO(png))
    W, H = img.size
    hit = nodes.merge(res, on="kegg_id")
    m = symmetric_limit(res["log2FC"].values, limit)
    fig = go.Figure()
    fig.add_layout_image(dict(source=img, xref="x", yref="y", x=0, y=0, sizex=W, sizey=H, sizing="stretch",
                              layer="below"))
    if not hit.empty:
        sig_mask = hit["q"] <= vres["q_thr"]
        fig.add_trace(go.Scatter(
            x=hit["x"], y=hit["y"], mode="markers",
            marker=dict(size=size, color=hit["log2FC"], colorscale=resolve_colormap(cmap, scale(pal["div"])),
                        cmin=-m, cmax=m, cmid=0, colorbar=dict(title=f"log2FC<br>{vres['treat']}/{vres['control']}"),
                        line=dict(width=np.where(sig_mask, 3, 1), color=np.where(sig_mask, "#0b0b0b", "#898781"))),
            text=hit["化合物"], customdata=np.stack([hit["kegg_id"], hit["FC"], hit["q"]], axis=-1),
            hovertemplate="%{text} (%{customdata[0]})<br>log2FC %{marker.color:.3f} (FC %{customdata[1]:.3g})"
                          "<br>補正後 p %{customdata[2]:.3g}<extra></extra>",
        ))
    fig.update_xaxes(range=[0, W], visible=False)
    fig.update_yaxes(range=[H, 0], visible=False, scaleanchor="x")
    fig.update_layout(width=W, height=H, margin=dict(l=0, r=0, t=30, b=0), title=f"{pid} {title}")
    show_chart(st, fig, width="content")
    st.caption(f"丸の色 = log2FC ({vres['treat']} / {vres['control']}, {vres['group']})、太い黒枠 = 補正後 p ≤ {vres['q_thr']}。"
               f" 経路図 © Kanehisa Laboratories. 詳細: https://www.kegg.jp/pathway/{pid}"
               " 論文などに経路図を掲載する場合は KEGG の利用条件 (https://www.kegg.jp/kegg/legal.html) を確認してください。")
    st.dataframe(hit[["化合物", "kegg_id", "log2FC", "FC", "p", "q"]].drop_duplicates("化合物").round(5),
                 hide_index=True)
