"""GUI (app.py) で使う Plotly の図。"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage

from lcms_analysis import CPD_COL, VALUE_COL, natural_key, pairwise_distances
from make_conc_table import FILE_COL, LABEL_COL

# 配色 (カテゴリ: 固定順で割り当て / 連続: 青 1 色 / 発散: 青-灰-赤)
PALETTE = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "div": ["#104281", "#5598e7", "#f0efec", "#e66767", "#a61e1e"],
        "line": "#52514e",
        "ink": "#0b0b0b",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "div": ["#6da7ec", "#256abf", "#383835", "#c43d3d", "#f08c8c"],
        "line": "#c3c2b7",
        "ink": "#ffffff",
    },
}


def scale(colors):
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


# ヒートマップで選べるカラーマップ。"既定" はこのアプリの配色 (連続 = 青 / 発散 = 青-灰-赤)。
# YlPu は Plotly / matplotlib に無いので、黄 -> ピンク -> 紫 の配色を独自に定義している。
CUSTOM_SCALES = {
    "YlPu": ["#ffffcc", "#fee8b0", "#fcc9b4", "#f9a3bd", "#e47bb5", "#bf52aa", "#8a2b97", "#4d0a6d"],
}
SEQUENTIAL_SCALES = ["YlPu", "Viridis", "Plasma", "Magma", "Inferno", "Cividis", "YlGnBu", "YlOrRd", "BuPu",
                     "RdPu", "PuBu", "Purples", "Blues", "Greys"]
DIVERGING_SCALES = ["RdBu", "PuOr", "PiYG", "PRGn", "BrBG", "RdYlBu", "Spectral", "Balance"]
AUTO_SCALE = "既定"


def colormap_options():
    names = [AUTO_SCALE]
    for n in SEQUENTIAL_SCALES + DIVERGING_SCALES:
        names += [n, f"{n}_r"]
    return names


def resolve_colormap(name, default):
    """カラーマップ名 (末尾 _r で反転) -> Plotly の colorscale。"既定" なら default を返す。"""
    if name == AUTO_SCALE:
        return default
    base, rev = (name[:-2], True) if name.endswith("_r") else (name, False)
    if base in CUSTOM_SCALES:
        colors = CUSTOM_SCALES[base]
        return scale(colors[::-1] if rev else colors)
    return px.colors.get_colorscale(base.lower() + ("_r" if rev else ""))


def color_map(names, pal):
    """名前 -> 色 を固定順で割り当てる (8 色を超えたら呼び出し側で警告する)。"""
    return {n: pal["series"][i % len(pal["series"])] for i, n in enumerate(names)}


def ref_line(fig, y):
    if y is not None:
        fig.add_hline(y=y, line=dict(color="#898781", width=1, dash="dot"), row="all", col="all")


# ---------------------------------------------------------------- 棒グラフ
def bar_samples(long, labels, color_col, color_order, pal, ytitle, ref=None):
    """サンプルごとの棒グラフ (x = label, 色 = 希釈など)。化合物ごとにパネルを分け、各パネルにサンプル名を表示する。"""
    compounds = long[CPD_COL].unique().tolist()
    n = len(compounds)
    panel, gap = 240, 110  # 1 パネルの高さ / パネル間 (サンプル名とタイトルの分) の px
    height = n * panel + (n - 1) * gap + 140
    fig = make_subplots(rows=n, cols=1, subplot_titles=compounds, vertical_spacing=gap / height if n > 1 else 0)
    colors = color_map(color_order, pal)
    for r, cpd in enumerate(compounds, start=1):
        d = long[long[CPD_COL] == cpd]
        for g in color_order:
            e = d[d[color_col] == g]
            if e.empty:
                continue
            fig.add_trace(go.Bar(
                x=e[LABEL_COL], y=e[VALUE_COL], name=g, legendgroup=g, showlegend=r == 1, marker_color=colors[g],
                customdata=np.stack([e[FILE_COL], e["condition"]], axis=-1),
                hovertemplate="%{x}<br>%{y:.4g}<br>%{customdata[1]}<br>%{customdata[0]}<extra>" + g + "</extra>",
            ), row=r, col=1)
        fig.update_xaxes(categoryorder="array", categoryarray=labels, tickangle=-60, tickfont_size=10, row=r, col=1)
        fig.update_yaxes(title=ytitle if n == 1 else None, row=r, col=1)
    for a in fig.layout.annotations:  # パネルのタイトル (化合物名) を左寄せに
        a.update(x=0, xanchor="left", font_size=13)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, barcornerradius=4, height=max(380, height),
                      legend_title_text=color_col, margin=dict(t=60))
    ref_line(fig, ref)
    return fig


def bar_summary(summ, points, x_col, x_order, color_col, color_order, pal, ytitle, stat_name,
                show_points=True, ref=None):
    """要約統計の棒グラフ (棒 = 平均/中央値, 誤差棒 = SD/IQR, 点 = 各サンプル)。"""
    compounds = list(dict.fromkeys(summ[CPD_COL]))
    colors = color_map(color_order, pal)
    fig = make_subplots(rows=len(compounds), cols=1, subplot_titles=compounds, vertical_spacing=0.3 / max(1, len(compounds)))
    for r, cpd in enumerate(compounds, start=1):
        s_c = summ[summ[CPD_COL] == cpd]
        p_c = points[points[CPD_COL] == cpd]
        for g in color_order:
            s = s_c[s_c[color_col] == g].set_index(x_col).reindex(x_order).reset_index()
            if s["n"].fillna(0).sum() == 0:
                continue
            fig.add_trace(go.Bar(
                x=s[x_col], y=s["center"], name=g, legendgroup=g, showlegend=r == 1, offsetgroup=g,
                marker_color=colors[g],
                error_y=dict(type="data", symmetric=False, array=s["upper"] - s["center"],
                             arrayminus=s["center"] - s["lower"], color=pal["ink"], thickness=1.2, width=4),
                customdata=np.stack([s["n"].fillna(0), s["lower"], s["upper"]], axis=-1),
                hovertemplate=f"%{{x}}<br>{g}<br>{stat_name}: %{{y:.3g}}<br>"
                              "範囲: %{customdata[1]:.3g} – %{customdata[2]:.3g}<br>n = %{customdata[0]}<extra></extra>",
            ), row=r, col=1)
            if show_points:
                p = p_c[p_c[color_col] == g]
                fig.add_trace(go.Scatter(
                    x=p[x_col], y=p[VALUE_COL], mode="markers", legendgroup=g, showlegend=False, offsetgroup=g,
                    marker=dict(size=8, color=colors[g], opacity=0.85, line=dict(width=1, color=pal["ink"])),
                    text=p[LABEL_COL], hovertemplate="%{text}<br>%{y:.3g}<extra></extra>",
                ), row=r, col=1)
        fig.update_xaxes(categoryorder="array", categoryarray=x_order, row=r, col=1)
    fig.update_layout(
        barmode="group", scattermode="group", bargap=0.3, bargroupgap=0.1, barcornerradius=4,
        legend_title_text=color_col, height=max(400, 300 * len(compounds)),
    )
    fig.update_yaxes(title=ytitle)
    ref_line(fig, ref)
    return fig


# ---------------------------------------------------------------- ヒートマップ
def heatmap_grid(mats, cscale, colorbar_title, zmid=None, zrange=None, hover=None):
    """複数パネルのヒートマップを横に並べる (行 = 化合物, 列 = mats の各行)。

    mats: {パネル名: DataFrame (行 = サンプル or 条件, 列 = 化合物)}
    hover: {パネル名: DataFrame (mats と同形の文字列)} ツールチップに追記する情報
    """
    names = list(mats)
    rows = sorted(set().union(*[m.columns for m in mats.values()]), key=natural_key)
    fig = make_subplots(
        rows=1, cols=len(names), shared_yaxes=True, subplot_titles=names, horizontal_spacing=0.02,
        column_widths=[max(1, len(mats[n])) for n in names],
    )
    for i, n in enumerate(names, start=1):
        m = mats[n].reindex(columns=rows).T
        extra = hover[n].reindex(columns=rows).T.fillna("").values if hover else np.full(m.shape, "")
        fig.add_trace(go.Heatmap(
            z=m.values, x=m.columns, y=m.index, coloraxis="coloraxis", xgap=1, ygap=1, customdata=extra,
            hovertemplate="%{x}<br>%{y}<br>%{z:.3g}%{customdata}<extra>" + n + "</extra>",
        ), row=1, col=i)
    fig.update_yaxes(autorange="reversed", tickfont_size=10)
    fig.update_xaxes(tickangle=-90, tickfont_size=10, type="category")
    coloraxis = dict(colorscale=cscale, colorbar_title=colorbar_title)
    if zmid is not None:
        coloraxis["cmid"] = zmid
    if zrange is not None:
        coloraxis["cmin"], coloraxis["cmax"] = zrange
    fig.update_layout(coloraxis=coloraxis, height=max(500, 14 * len(rows) + 180), margin=dict(t=40, l=220))
    return fig


def clustered_heatmap(X, method, metric, cscale, zmid, pal, title, p=3):
    """階層的クラスタリング (サンプル・化合物の両方) + デンドログラム付きヒートマップ。"""
    dist_cols = pairwise_distances(X.values, metric, p)                 # サンプル
    dist_rows = pairwise_distances(X.values.T, metric, p)               # 化合物
    if not (np.isfinite(dist_cols).all() and np.isfinite(dist_rows).all()):
        raise ValueError("距離に NaN / inf が含まれます (分散 0 の行がある、負の値に非対応の距離など)")
    col_link = linkage(dist_cols, method=method)
    row_link = linkage(dist_rows, method=method)
    dc = dendrogram(col_link, no_plot=True, labels=X.index.tolist())
    dr = dendrogram(row_link, no_plot=True, labels=X.columns.tolist())
    M = X.iloc[dc["leaves"], dr["leaves"]].T  # 行 = 化合物, 列 = サンプル
    xs = 5 + 10 * np.arange(M.shape[1])       # scipy の葉座標 (5, 15, 25, ...) に合わせる
    ys = 5 + 10 * np.arange(M.shape[0])

    fig = make_subplots(
        rows=2, cols=2, column_widths=[0.12, 0.88], row_heights=[0.12, 0.88],
        shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.005, vertical_spacing=0.005,
    )
    line = dict(color=pal["line"], width=1)
    for xi, yi in zip(dc["icoord"], dc["dcoord"]):
        fig.add_trace(go.Scatter(x=xi, y=yi, mode="lines", line=line, hoverinfo="skip"), row=1, col=2)
    for xi, yi in zip(dr["icoord"], dr["dcoord"]):
        fig.add_trace(go.Scatter(x=[-v for v in yi], y=xi, mode="lines", line=line, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Heatmap(
        z=M.values, x=xs, y=ys, customdata=np.dstack(np.meshgrid(M.columns, M.index)),
        colorscale=cscale, zmid=zmid, xgap=1, ygap=1,
        colorbar=dict(orientation="h", x=0.06, xanchor="center", y=0.94, len=0.11, thickness=10),
        hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>%{z:.3g}<extra></extra>",
    ), row=2, col=2)

    fig.update_xaxes(tickvals=xs, ticktext=M.columns, tickangle=-90, tickfont_size=10, row=2, col=2)
    fig.update_yaxes(tickvals=ys, ticktext=M.index, tickfont_size=10, side="right", row=2, col=2)
    for r, c in [(1, 1), (1, 2), (2, 1)]:
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
    fig.update_yaxes(range=[0, 10 * M.shape[0]], row=2)
    fig.update_yaxes(range=[0, max(max(d) for d in dc["dcoord"]) * 1.05], row=1)
    fig.update_xaxes(range=[-max(max(d) for d in dr["dcoord"]) * 1.05, 0], col=1)
    fig.update_yaxes(showticklabels=True, row=2, col=2)
    fig.update_xaxes(range=[0, 10 * M.shape[1]], col=2)
    fig.update_layout(
        title=title, showlegend=False, height=max(600, 14 * M.shape[0] + 260),
        margin=dict(t=50, r=200),
    )
    return fig, M


# ---------------------------------------------------------------- 散布図 (PCA / UMAP)
def scatter_2d(coords, xcol, ycol, show_text, title, xtitle, ytitle, pal, groups=None, group_order=None):
    """groups (condition など) があれば色分けし、無ければ 1 色で描く。"""
    fig = go.Figure()
    if groups is None:
        sets = [(None, coords, pal["series"][0])]
    else:
        cmap = color_map(group_order, pal)
        sets = [(g, coords[groups == g], cmap[g]) for g in group_order if (groups == g).any()]
    for g, c, color in sets:
        fig.add_trace(go.Scatter(
            x=c[xcol], y=c[ycol], mode="markers+text" if show_text else "markers", name=g or "",
            showlegend=g is not None, text=c.index, textposition="top center", textfont=dict(size=10),
            marker=dict(size=10, color=color, line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
            hovertemplate="%{text}<br>" + (f"{g}<br>" if g else "") + xtitle + ": %{x:.3g}<br>"
                          + ytitle + ": %{y:.3g}<extra></extra>",
        ))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle, height=480)
    return fig


# ---------------------------------------------------------------- 群ごとの分布 (検定と一緒に載せる図)
PLOT_AUTO = "自動 (検定に合わせる)"
PLOT_BOX, PLOT_BAR_SD, PLOT_BAR_SEM, PLOT_VIOLIN, PLOT_DOT = (
    "箱ひげ図 (中央値・四分位)", "棒グラフ (平均 ± SD)", "棒グラフ (平均 ± SEM)", "バイオリンプロット", "点 + 平均 ± SD")
PLOT_KINDS = [PLOT_AUTO, PLOT_BOX, PLOT_BAR_SD, PLOT_BAR_SEM, PLOT_VIOLIN, PLOT_DOT]


def resolve_plot_kind(kind, parametric):
    """自動なら、平均を比べる検定 (t 検定・ANOVA) は 平均 ± SD の棒、順位の検定は箱ひげ図にする。"""
    if kind != PLOT_AUTO:
        return kind
    return PLOT_BAR_SD if parametric else PLOT_BOX


def group_plot(df, value, group_col, groups, label_col, pal, kind, show_points=True, logy=False):
    """群 (condition など) ごとの分布の図。群 i は x = i の位置に置き、目盛りに群名を表示する
    (有意差の括弧などを同じ座標で重ねられるように)。"""
    cmap = color_map(groups, pal)
    rng = np.random.default_rng(0)  # 点の横方向のずらしを再現できるよう乱数の種を固定
    fig = go.Figure()
    for i, g in enumerate(groups):
        d = df[df[group_col] == g].dropna(subset=[value])
        y, txt, c = d[value].astype(float).values, d[label_col].values, cmap[g]
        xs = np.full(len(y), i, dtype=float)
        hover = "%{text}<br>%{y:.4g}<extra>" + str(g) + "</extra>"
        if kind == PLOT_BOX:
            fig.add_trace(go.Box(y=y, x=xs, name=g, marker_color=c, line_width=1.5, text=txt, width=0.55,
                                 boxpoints="all" if show_points else "outliers", jitter=0.4, pointpos=0,
                                 showlegend=False, hovertemplate=hover))
            continue
        if kind == PLOT_VIOLIN:
            rgb = tuple(int(c.lstrip("#")[k:k + 2], 16) for k in (0, 2, 4))
            fig.add_trace(go.Violin(
                y=y, x=xs, name=g, line_color=c, fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.35)", width=0.8,
                spanmode="hard", box=dict(visible=True, width=0.25, fillcolor="rgba(255,255,255,0.8)"),
                meanline_visible=True, points="all" if show_points else False, jitter=0.3, pointpos=0,
                marker=dict(size=7, color=c, line=dict(width=1, color="white")), text=txt, showlegend=False,
                hovertemplate=hover))
            continue
        m = y.mean() if len(y) else np.nan
        sd = y.std(ddof=1) if len(y) > 1 else np.nan
        err = sd / np.sqrt(len(y)) if kind == PLOT_BAR_SEM else sd
        name = "SEM" if kind == PLOT_BAR_SEM else "SD"
        ebar = dict(type="data", array=[err], color=pal["ink"], thickness=1.5, width=8)
        tip = f"{g}<br>平均 %{{y:.4g}}<br>{name} {err:.4g}<br>n = {len(y)}<extra></extra>"
        if kind in (PLOT_BAR_SD, PLOT_BAR_SEM):
            fig.add_trace(go.Bar(x=[i], y=[m], width=0.6, marker_color=c, opacity=0.85, error_y=ebar,
                                 showlegend=False, hovertemplate=tip))
        else:  # 点 + 平均 ± SD
            fig.add_trace(go.Scatter(x=[i], y=[m], mode="markers", error_y=ebar, showlegend=False, hovertemplate=tip,
                                     marker=dict(symbol="line-ew", size=34, line=dict(width=3, color=pal["ink"]))))
        if show_points or kind == PLOT_DOT:
            fig.add_trace(go.Scatter(
                x=xs + (rng.random(len(y)) - 0.5) * 0.35, y=y, mode="markers", text=txt, showlegend=False,
                marker=dict(size=8, color=c, line=dict(width=1, color="rgba(255,255,255,0.9)")), hovertemplate=hover))
    fig.update_xaxes(tickvals=list(range(len(groups))), ticktext=list(groups), range=[-0.6, len(groups) - 0.4],
                     zeroline=False)
    fig.update_yaxes(type="log" if logy else "linear")
    return fig
