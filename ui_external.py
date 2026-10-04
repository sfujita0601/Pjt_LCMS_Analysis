"""タブ: 外部データの取り込み / 変数グループ / 正準相関分析。"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lcms_analysis as la
import lcms_external as lx
import lcms_stats as ls
from lcms_analysis import COND_COL, GROUP_COL
from lcms_plots import color_map, resolve_colormap, scale
from make_conc_table import LABEL_COL, is_std
from ui_common import colormap_select, current_editor, reset_editor, show_chart, stable_editor, to_csv_bytes
from ui_project import sha256

GROUPS_KEY = "var_groups"


# ---------------------------------------------------------------- 外部データ
class _LocalFile:
    """data/ にあるファイルを file_uploader の戻り値と同じように扱う。"""

    def __init__(self, path):
        self.name, self._path = path.name, path

    def getvalue(self):
        return self._path.read_bytes()


def external_tab(labels, harmonize, data_dir):
    """外部データのファイルを読み込み、設定を返す: [dict(name, wide, mode, sha256, columns)]"""
    st.caption("サンプル ID ごとに複数の指標を持つ表 (脂質・臨床検査値・体重など) を読み込み、label と照合して統合します。"
               "1 行 = 1 サンプル、1 列 = 1 指標 の形式 (CSV / TSV / Excel) にしてください。"
               "統合した指標は正規化 (IS 補正・PQN など) の対象外で、相関・クラスタリング・PCA・検定などで化合物と同様に使えます。")
    files = st.file_uploader("外部データ (CSV / TSV / Excel, 複数可)", type=["csv", "tsv", "txt", "xlsx", "xlsm"],
                             accept_multiple_files=True, key="_up_ext")
    local = sorted(p for p in data_dir.glob("*") if p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xlsm")
                   and not p.stem.endswith(("_conc", "_conc_IS")))
    picks = st.multiselect("または data/ から選択", local, format_func=lambda p: p.name, key="ext_local")
    files = list(files) + [_LocalFile(p) for p in picks]
    loaded = st.session_state.get("_project_loaded") or {}
    expected = {d["name"]: d["sha256"] for d in loaded.get("external", [])}
    if expected:
        now = {f.name: sha256(f.getvalue()) for f in files}
        missing = [n for n in expected if n not in now]
        changed = [n for n in expected if n in now and now[n] != expected[n]]
        if missing or changed:
            st.warning("読み込んだ解析設定で使われていた外部データ: "
                       + (f"未選択 {', '.join(missing)}" if missing else "")
                       + (f" / 内容が異なる {', '.join(changed)}" if changed else "")
                       + "。同じファイルを選ぶと設定が復元されます。")
    specs = []
    sample_labels = [lb for lb in labels if not is_std(lb)]
    for f in files:
        key = f"ext::{f.name}"
        raw = f.getvalue()
        with st.expander(f.name, expanded=True):
            try:
                df, sheets = lx.read_table(raw, f.name)
                if sheets:
                    sheet = st.selectbox("シート", sheets, key=f"{key}::sheet")
                    df, _ = lx.read_table(raw, f.name, sheet)
            except Exception as e:
                st.error(f"読み込めません: {e}")
                continue
            if df.empty or len(df.columns) < 2:
                st.error("ID 列と指標の列が必要です")
                continue
            c1, c2, c3 = st.columns(3)
            id_col = c1.selectbox("サンプル ID の列", list(df.columns), key=f"{key}::id")
            mode = c2.selectbox("照合方法", [lx.MATCH_EXACT, lx.MATCH_DIGITS], key=f"{key}::mode")
            prefix = c3.text_input("列名の接頭辞", value=Path(f.name).stem, key=f"{key}::prefix",
                                   help="LC-MS の化合物名と区別するため「接頭辞: 列名」とします。空欄なら列名のまま")
            num = lx.numeric_columns(df, id_col)
            cols = st.multiselect("使う指標 (数値の列)", num, default=num, key=f"{key}::cols")
            if not cols:
                st.warning("指標を 1 つ以上選んでください")
                continue
            wide, dup = lx.prepare(df, id_col, cols, prefix.strip(), mode, harmonize)
            keys = {lb: lx.match_key(lb, mode, harmonize) for lb in sample_labels}
            matched = [lb for lb, k in keys.items() if k in wide.index]
            not_found = [lb for lb, k in keys.items() if k not in wide.index]
            extra = sorted(set(wide.index) - set(keys.values()))
            st.markdown(f"一致したサンプル: **{len(matched)} / {len(sample_labels)}** ・ 指標 {len(cols)} 個"
                        + (f" ・ 重複 ID {dup} 行は平均しました" if dup else ""))
            if not_found:
                st.caption("外部データに見つからない label: " + ", ".join(not_found))
            if extra:
                st.caption("LC-MS 側に無い外部データの ID: " + ", ".join(map(str, extra[:30])) + (" ..." if len(extra) > 30 else ""))
            st.dataframe(df.head(10), hide_index=True, height=200)
            specs.append(dict(name=f.name, wide=wide, mode=mode, sha256=sha256(raw), raw=raw))
    return specs


def merge_external(base, specs, harmonize, existing):
    """外部データの列を base に結合する。({ファイル名: 追加した列}, 結合後の表)"""
    groups = {}
    for spec in specs:
        wide = spec["wide"].copy()
        # 既存の列名 (化合物・比・ほかの外部データ) と重なる場合は番号を付ける
        wide.columns = [c if c not in existing and c not in base.columns else f"{c} ({i})"
                        for i, c in enumerate(wide.columns, start=2)]
        base, _, _ = lx.merge_into(base, LABEL_COL, wide, spec["mode"], harmonize)
        cols = [c for c in wide.columns if base[c].notna().any()]
        groups[spec["name"]] = cols
    return base, groups


# ---------------------------------------------------------------- 変数グループ
def groups_tab(variables, auto_groups):
    """変数 (化合物・比・外部指標) のグループを定義する。{グループ名: [変数]} を返す。

    auto_groups: 自動で用意するグループ (外部データのファイルごと、比など) {名前: [変数]}
    """
    init = pd.DataFrame({"名前": pd.Series(dtype=str), "変数": pd.Series(dtype=str)})
    st.caption("化合物・比・外部データの指標をグループにまとめます。グループは 正準相関 タブのほか、"
               "各タブの「対象の化合物」でまとめて選ぶときにも使えます。変数は「; 」で区切ります。")

    def append(rows):
        cur = current_editor(GROUPS_KEY)
        cur = init if cur is None else cur
        have = set(cur["名前"].dropna())
        add = pd.DataFrame([r for r in rows if r["名前"] not in have])
        reset_editor(GROUPS_KEY, pd.concat([cur, add], ignore_index=True))
        st.rerun()

    c1, c2, c3 = st.columns(3)
    if c1.button("よく使う分類を追加", key="_btn_grp_preset",
                 help="アミノ酸・BCAA・有機酸・核酸関連など (データにある化合物が 2 つ以上あるもの)"):
        rows = []
        for name, members in la.GROUP_PRESETS.items():
            m = [x for x in members if x in variables]
            if len(m) >= 2:
                rows.append({"名前": name, "変数": la.GROUP_SEP.join(m)})
        append(rows)
    if c2.button("外部データ・比をグループに追加", key="_btn_grp_auto", disabled=not auto_groups):
        append([{"名前": n, "変数": la.GROUP_SEP.join(m)} for n, m in auto_groups.items() if m])
    with st.form("grp_add", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 5, 1])
        name = c1.text_input("グループ名", key="_grp_name")
        members = c2.multiselect("変数", variables, key="_grp_members")
        if c3.form_submit_button("追加") and name and members:
            append([{"名前": name, "変数": la.GROUP_SEP.join(members)}])
    edited = stable_editor(GROUPS_KEY, init, num_rows="dynamic", hide_index=True, width="stretch",
                           column_config={"変数": st.column_config.TextColumn("変数", width="large")})
    groups, missing = {}, []
    for _, r in edited.iterrows():
        name = str(r.get("名前") or "").strip()
        if not name:
            continue
        m = la.parse_members(r.get("変数"))
        missing += [f"{name}: {x}" for x in m if x not in variables]
        m = [x for x in m if x in variables]
        if m:
            groups[name] = m
    if missing:
        st.caption("データに無いため除外した変数: " + ", ".join(missing))
    if groups:
        st.dataframe(pd.DataFrame({"グループ": list(groups), "変数の数": [len(v) for v in groups.values()]}),
                     hide_index=True)
    st.session_state["_var_groups"] = groups  # compound_picker から参照する ("_" で始まるので設定保存の対象外)
    return groups


# ---------------------------------------------------------------- 正準相関分析
@st.cache_data(show_spinner="正準相関分析を計算中...")
def _cca(X, Y, reg, n_perm, seed):
    return ls.cca(X, Y, reg, n_perm, seed)


def _block(sub, cols, log, max_missing):
    X = sub[cols].astype(float)
    X = X.loc[:, X.notna().mean() >= 1 - max_missing]
    # 欠損の補完: 正の値だけの列 (濃度など) は最小値の 1/2、負の値を含む列は中央値
    positive = (X > 0).where(X.notna(), True).all()
    X = X.fillna(X.min().where(positive, np.nan) / 2).fillna(X.median())
    if log:
        X = la.log10_safe(X)
    sd = X.std()
    return X.loc[:, sd > 0]


def cca_tab(base, groups, dil_groups, conds, pal):
    st.caption("2 つの変数グループ (例: アミノ酸 と 脂質の指標) の間で、互いに最も相関が高くなる線形結合 (正準変量) を求めます"
               " (Hotelling, 1936)。変数の数がサンプル数に比べて多いと正準相関は見かけ上 1 に近づくので、"
               "正則化・並べ替え検定・一つ抜き交差検証の結果も合わせて判断してください。")
    if len(groups) < 2:
        st.info("変数グループ タブでグループを 2 つ以上作ってください。")
        return
    names = list(groups)
    c1, c2, c3, c4 = st.columns(4)
    gx = c1.selectbox("グループ X", names, index=0, key="cca_x")
    gy = c2.selectbox("グループ Y", names, index=1, key="cca_y")
    dil = c3.selectbox("希釈グループ", dil_groups, key="cca_dil")
    use = c4.multiselect("使う condition (空欄 = 全サンプル)", conds, key="cca_conds")
    c1, c2, c3, c4, c5 = st.columns(5)
    log_x = c1.checkbox("X を log10 変換", value=True, key="cca_logx")
    log_y = c2.checkbox("Y を log10 変換", value=True, key="cca_logy",
                        help="負の値を含む列は変換しません")
    reg = c3.slider("正則化 (0 = 古典的 CCA)", 0.0, 0.95, 0.5, 0.05, key="cca_reg",
                    help="共分散を単位行列に近づける強さ (canonical ridge)。変数がサンプル数より多いときは必須")
    n_perm = c4.selectbox("並べ替え検定の回数", [0, 199, 499, 999, 1999], index=3, key="cca_perm")
    max_missing = c5.slider("欠損率がこれ以下の変数を使用", 0.0, 1.0, 0.5, 0.05, key="cca_miss")

    if gx == gy:
        st.warning("異なるグループを選んでください")
        return
    overlap = set(groups[gx]) & set(groups[gy])
    if overlap:
        st.warning("X と Y の両方に含まれる変数は Y から除外しました: " + ", ".join(sorted(overlap)))
    sub = base[(base[GROUP_COL] == dil) & ~base[LABEL_COL].map(is_std)]
    if use:
        sub = sub[sub[COND_COL].isin(use)]
    xcols = [c for c in groups[gx] if c in sub]
    ycols = [c for c in groups[gy] if c in sub and c not in overlap]
    if not xcols or not ycols:
        st.warning("重なりを除くと変数が残らないグループがあります。重ならないグループを選んでください。")
        return
    # どちらかのグループの値がほとんど無いサンプル (外部データと照合できなかったサンプルなど) は除く
    keep = (sub[xcols].notna().mean(axis=1) >= 0.5) & (sub[ycols].notna().mean(axis=1) >= 0.5)
    dropped = sorted(sub.loc[~keep, LABEL_COL])
    sub = sub[keep]
    X, Y = _block(sub, xcols, log_x, max_missing), _block(sub, ycols, log_y, max_missing)
    n, p, q = len(sub), X.shape[1], Y.shape[1]
    if n < 5 or p < 1 or q < 1:
        st.warning(f"サンプルまたは変数が足りません (サンプル {n}, X {p}, Y {q})")
        return
    st.markdown(f"サンプル **{n}** ・ X: {gx} **{p}** 変数 ・ Y: {gy} **{q}** 変数"
                + (f" ・ 除外したサンプル: {', '.join(dropped)}" if dropped else ""))
    if reg == 0 and p + q >= n - 1:
        st.warning(f"変数の数 (X {p} + Y {q}) がサンプル数 ({n}) に対して多すぎるため、古典的 CCA では正準相関が自明に 1 になります。"
                   "正則化を 0 より大きくしてください。")
    res = _cca(X.values, Y.values, reg, int(n_perm), 0)
    k = len(res["r"])

    st.subheader("正準相関")
    tbl = pd.DataFrame({"成分": [f"CV{i + 1}" for i in range(k)], "正準相関 (当てはめ)": res["r"],
                        "並べ替え検定 p": res["p"] if n_perm else np.nan})
    c1, c2 = st.columns([1, 2])
    c1.dataframe(tbl.head(10).round(4), hide_index=True)
    c1.markdown(f"第 1 成分の一つ抜き交差検証の相関: **{res['r_loo']:.3f}**")
    c1.caption("当てはめの正準相関は過大になりやすく、交差検証の相関は新しいサンプルでどの程度再現されるかの目安です。"
               "並べ替え検定の p は、X と Y の対応をランダムに入れ替えたときにその成分の正準相関がこれ以上になる割合です。")
    fig = go.Figure(go.Bar(x=tbl["成分"][:10], y=tbl["正準相関 (当てはめ)"][:10], marker_color=pal["series"][0],
                           text=[f"p={v:.3f}" if pd.notna(v) else "" for v in tbl["並べ替え検定 p"][:10]],
                           textposition="outside", hovertemplate="%{x}: r = %{y:.3f}<extra></extra>"))
    fig.update_layout(height=320, yaxis=dict(range=[0, 1.1], title="正準相関"), margin=dict(t=20), barcornerradius=4)
    show_chart(c2, fig, width="stretch")

    st.subheader("正準変量の散布図")
    c1, c2 = st.columns([1, 1])
    comp = c1.selectbox("成分", list(range(1, k + 1)), format_func=lambda i: f"CV{i}", key="cca_comp")
    color_by = c2.radio("色分け", ["condition", "なし"], horizontal=True, key="cca_color")
    sc = pd.DataFrame({"u": res["U"][:, comp - 1], "v": res["V"][:, comp - 1], LABEL_COL: sub[LABEL_COL].values,
                       COND_COL: sub[COND_COL].replace("", "未設定").values})
    cats = [c for c in conds + ["未設定"] if (sc[COND_COL] == c).any()] if color_by == "condition" else ["全サンプル"]
    cmap = color_map(cats, pal)
    fig = go.Figure()
    for c in cats:
        g = sc if color_by == "なし" else sc[sc[COND_COL] == c]
        fig.add_trace(go.Scatter(x=g["u"], y=g["v"], mode="markers+text", name=c, text=g[LABEL_COL],
                                 textposition="top center", textfont=dict(size=10),
                                 marker=dict(size=10, color=cmap[c], line=dict(width=1.5, color="white")),
                                 hovertemplate="%{text}<br>X 側 %{x:.3f}<br>Y 側 %{y:.3f}<extra>" + c + "</extra>"))
    fig.update_layout(height=460, xaxis_title=f"{gx} の正準変量 CV{comp}", yaxis_title=f"{gy} の正準変量 CV{comp}",
                      title=f"CV{comp}: r = {res['r'][comp - 1]:.3f}")
    show_chart(st, fig, width="stretch")

    st.subheader("変数の寄与 (構造相関)")
    lx_, ly_ = pd.DataFrame(res["load_x"], index=X.columns), pd.DataFrame(res["load_y"], index=Y.columns)
    if k >= 2:
        fig = go.Figure()
        t = np.linspace(0, 2 * np.pi, 200)
        for rad, dash in [(1, "solid"), (0.5, "dot")]:
            fig.add_trace(go.Scatter(x=rad * np.cos(t), y=rad * np.sin(t), mode="lines", hoverinfo="skip",
                                     line=dict(color="#c3c2b7", width=1, dash=dash), showlegend=False))
        for name, L, color in [(gx, lx_, pal["series"][0]), (gy, ly_, pal["series"][1])]:
            fig.add_trace(go.Scatter(x=L[0], y=L[1], mode="markers+text", name=name, text=L.index,
                                     textposition="top center", textfont=dict(size=9),
                                     marker=dict(size=8, color=color),
                                     hovertemplate="%{text}<br>CV1 %{x:.3f}<br>CV2 %{y:.3f}<extra>" + name + "</extra>"))
        fig.update_xaxes(range=[-1.1, 1.1], title="正準変量 CV1 との相関", zeroline=True)
        fig.update_yaxes(range=[-1.1, 1.1], title="正準変量 CV2 との相関", zeroline=True, scaleanchor="x")
        fig.update_layout(height=620, title="相関円 (各変数と自分の側の正準変量との相関)")
        show_chart(st, fig, width="stretch")
        st.caption("外側の円の近くにある変数ほど、その成分への寄与が大きい。X と Y の変数が同じ方向にあれば正の関係。"
                   "点線の円は |相関| = 0.5。")
    load = pd.concat([
        pd.DataFrame({"グループ": gx, "変数": X.columns, "CV1 構造相関": res["load_x"][:, 0],
                      "CV1 交差構造相関": res["cross_x"][:, 0]}),
        pd.DataFrame({"グループ": gy, "変数": Y.columns, "CV1 構造相関": res["load_y"][:, 0],
                      "CV1 交差構造相関": res["cross_y"][:, 0]}),
    ]).sort_values("CV1 構造相関", key=np.abs, ascending=False)
    st.dataframe(load.round(4), hide_index=True, height=300)
    st.caption("交差構造相関 = 変数と相手側の正準変量との相関 (相手のグループの変動をどれだけ反映するか)。")
    st.download_button("構造相関 (CSV)", to_csv_bytes(load), "cca_loadings.csv", "text/csv", key="_dl_cca")

    st.subheader("X と Y の変数間の相関")
    c1, c2 = st.columns([1, 3])
    method = c1.selectbox("相関係数", ["Spearman", "Pearson"], key="cca_cor_method")
    with c1:
        cmap_name = colormap_select("cca_cmap")
    Rxy = pd.DataFrame([[(X[a].corr(Y[b], method=method.lower())) for b in Y.columns] for a in X.columns],
                       index=X.columns, columns=Y.columns)
    fig = go.Figure(go.Heatmap(z=Rxy.values, x=Rxy.columns, y=Rxy.index, zmin=-1, zmax=1, zmid=0, xgap=1, ygap=1,
                               colorscale=resolve_colormap(cmap_name, scale(pal["div"])), colorbar=dict(title="r"),
                               hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont_size=9)
    fig.update_xaxes(tickangle=-90, tickfont_size=9)
    fig.update_layout(height=max(400, 14 * p + 200), margin=dict(l=200, b=160, t=20))
    show_chart(c2, fig, width="stretch")
    c2.download_button("相関行列 (CSV)", to_csv_bytes(Rxy, index=True), "cca_cross_correlation.csv", "text/csv",
                       key="_dl_cca_r")
