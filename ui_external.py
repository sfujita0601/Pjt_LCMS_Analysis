"""タブ: 外部データの取り込み / 変数グループ / 正準相関分析。"""
from pathlib import Path

import warnings

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
from ui_common import colormap_select, current_editor, register_output, reset_editor, show_chart, stable_editor, to_csv_bytes
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
            c1, c2 = st.columns(2)
            dup_rule = c1.radio("同じ ID が複数行ある場合", ["結合しない (エラーとして表示)", "平均する", "最初の行を使う"],
                                key=f"{key}::dup", help="一対多の対応で行が増えないよう、ID ごとに 1 行にしてから結合します")
            auto_sim = any(w in f.name.lower() for w in ("simulated", "demo", "mock")) or "模擬" in f.name
            is_sim = c2.checkbox("模擬データ (架空の値)", value=auto_sim, key=f"{key}::sim",
                                 help="模擬データの指標は、図と出力に「模擬データ」と表示します")
            keys_all = df[id_col].map(lambda v: lx.match_key(v, mode, harmonize))
            dup_ids = sorted(set(keys_all[keys_all.duplicated(keep=False)].dropna()))
            if dup_ids:
                st.warning(f"同じ ID が複数行: {', '.join(map(str, dup_ids[:10]))}" + (" ..." if len(dup_ids) > 10 else ""))
                if dup_rule.startswith("結合しない"):
                    st.error("重複 ID があるため、このファイルは結合しません。「同じ ID が複数行ある場合」を選んでください。")
                    continue
                if dup_rule == "最初の行を使う":
                    df = df[~keys_all.duplicated(keep="first")]
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
            specs.append(dict(name=f.name, wide=wide, mode=mode, sha256=sha256(raw), raw=raw, simulated=is_sim))
    return specs


def merge_external(base, specs, harmonize, existing):
    """外部データの列を base に結合する。({ファイル名: 追加した列}, 結合後の表)"""
    groups, sim = {}, set()
    for spec in specs:
        wide = spec["wide"].copy()
        # 既存の列名 (化合物・比・ほかの外部データ) と重なる場合は番号を付ける
        wide.columns = [c if c not in existing and c not in base.columns else f"{c} ({i})"
                        for i, c in enumerate(wide.columns, start=2)]
        # 個体ID (未入力なら label) で照合する。reindex なので行は増えない (一対多の結合は起こらない)
        key_col = "個体ID" if "個体ID" in base else LABEL_COL
        n_before = len(base)
        base, _, _ = lx.merge_into(base, key_col, wide, spec["mode"], harmonize)
        assert len(base) == n_before, "外部データの結合で行数が変わりました"
        cols = [c for c in wide.columns if base[c].notna().any()]
        groups[spec["name"]] = cols
        if spec.get("simulated"):
            sim |= set(cols)
    st.session_state["_simulated_cols"] = sim
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


# ---------------------------------------------------------------- 正準相関分析 (探索)
EXPLORE_NOTE = "学習データ内の探索結果 (独立に検証された関連・因果関係ではありません)"


def _simulated(cols):
    sim = st.session_state.get("_simulated_cols") or set()
    return sorted(set(cols) & sim)


def _mark(fig, simulated):
    """図に「探索結果」「模擬データ」の注記を入れる (保存した画像にも残るように)。"""
    text = EXPLORE_NOTE + (" ・ 模擬データを含む" if simulated else "")
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0, y=1.08, showarrow=False, xanchor="left",
                       font=dict(size=11, color="#a61e1e" if simulated else "#52514e"))
    return fig


def _prepare(sub, cols, max_missing):
    """変数の選択 (欠測率) と定数列の除外。補完・対数変換はしない (後で行う)。"""
    X = sub[cols].astype(float)
    dropped_missing = [c for c in X.columns if X[c].notna().mean() < 1 - max_missing]
    X = X.drop(columns=dropped_missing)
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    return X.drop(columns=const), dropped_missing, const


def _impute(X):
    positive = (X > 0).where(X.notna(), True).all()
    return X.fillna(X.min().where(positive, np.nan) / 2).fillna(X.median())


@st.cache_data(show_spinner="正準相関分析を計算中...")
def _cca_all(X, Y, units, reg, perm_mode, n_perm, strata, xnames, ynames, do_cv, Xraw, Yraw, log_x, log_y):
    core = ls.cca_core(X, Y, reg)
    diag = ls.cca_diagnostics(X, Y)
    pvals = ls.cca_permutation(X, Y, reg, n_perm, 0, strata if perm_mode == "condition 内で置換" else None) \
        if perm_mode != "行わない" and n_perm else None
    loo, stab = ls.cca_leave_one_out(X, Y, units, reg, 2, 5, list(xnames), list(ynames))
    cv = ls.cca_cross_validation(Xraw, Yraw, units, reg, log_x, log_y) if do_cv else None
    return core, diag, pvals, loo, stab, cv


def cca_tab(base, groups, dil_groups, conds, pal):
    st.caption("2 つの変数グループ (例: アミノ酸 と 脂質) の間で、共同して変動するパターン (正準モード) を探索します。"
               "小標本では学習データ内の正準相関が高くなりやすいので、負荷量・個別相関・除外再解析の安定性を合わせて見てください。")
    st.warning(EXPLORE_NOTE)
    if len(groups) < 2:
        st.info("変数グループ タブでグループを 2 つ以上作ってください。")
        return
    names = list(groups)
    c1, c2, c3, c4 = st.columns(4)
    gx = c1.selectbox("グループ X", names, index=0, key="cca_x")
    gy = c2.selectbox("グループ Y", names, index=1, key="cca_y")
    dil = c3.selectbox("希釈グループ", dil_groups, key="cca_dil")
    use = c4.multiselect("使う condition (空欄 = 全サンプル)", conds, key="cca_conds")
    c1, c2, c3, c4 = st.columns(4)
    log_x = c1.selectbox("X の対数変換", la.LOG_CHOICES, index=1, key="cca_logx_mode")
    log_y = c2.selectbox("Y の対数変換", la.LOG_CHOICES, index=1, key="cca_logy_mode", help="負の値を含む列は変換しません")
    reg = c3.slider("正則化 (0 = 古典的 CCA)", 0.0, 0.95, 0.5, 0.05, key="cca_reg",
                    help="共分散を単位行列に近づける強さ (canonical ridge)。変数がサンプル数に比べて多いときに推奨")
    max_missing = c4.slider("欠測率がこれ以下の変数を使用", 0.0, 1.0, 0.5, 0.05, key="cca_miss")
    c1, c2 = st.columns(2)
    missing_mode = c1.radio("欠測の扱い", ["完全ケースのみ (欠測のある試料を除く)", "補完 (最小値の 1/2)"], key="cca_missing",
                            horizontal=True)
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
        st.warning("重なりを除くと変数が残らないグループがあります。")
        return
    X0, dx_m, dx_c = _prepare(sub, xcols, max_missing)
    Y0, dy_m, dy_c = _prepare(sub, ycols, max_missing)
    if missing_mode.startswith("完全"):
        ok = X0.notna().all(axis=1) & Y0.notna().all(axis=1)
    else:
        ok = (X0.notna().mean(axis=1) >= 0.5) & (Y0.notna().mean(axis=1) >= 0.5)
    dropped_rows = sorted(sub.loc[~ok.values, LABEL_COL])
    sub, X0, Y0 = sub[ok.values], X0[ok.values], Y0[ok.values]
    # 補完 (正の値の列は最小値の 1/2、負の値を含む列は中央値) -> 対数変換 (正の値の列のみ)
    X = la.apply_log(_impute(X0), log_x)
    Y = la.apply_log(_impute(Y0), log_y)
    units = sub["個体ID"].values if "個体ID" in sub else sub[LABEL_COL].values
    n, p, q = len(sub), X.shape[1], Y.shape[1]
    simulated = _simulated(list(X.columns) + list(Y.columns))
    if simulated:
        st.error("模擬データ (架空の値) を含みます: " + ", ".join(simulated[:6]) + (" ..." if len(simulated) > 6 else ""))

    st.subheader("診断")
    if n < 5 or p < 1 or q < 1:
        st.warning(f"サンプルまたは変数が足りません (完全ケース {n}, X {p}, Y {q})")
        return
    repeated = len(pd.unique(units)) < len(units)
    diag_rows = {"完全ケース数 (行)": n, "独立した生物学的単位 (個体ID)": len(pd.unique(units)), "X の変数の数": p,
                 "Y の変数の数": q}
    st.caption("除いた変数 / 試料: " + " ・ ".join(filter(None, [
        f"欠測率で除外 X {len(dx_m)} / Y {len(dy_m)}" if dx_m or dy_m else "",
        f"定数列 {', '.join(dx_c + dy_c)}" if dx_c or dy_c else "",
        f"欠測で除いた試料 {', '.join(dropped_rows)}" if dropped_rows else "",
    ])) or "なし")
    if reg == 0 and p + q >= n - 1:
        st.warning(f"変数の数 (X {p} + Y {q}) がサンプル数 ({n}) に対して多く、古典的 CCA では正準相関が自明に 1 になります。")

    c1, c2, c3 = st.columns(3)
    perm_mode = c1.selectbox("置換検定", ["行わない", "全体で置換", "condition 内で置換"], key="cca_perm_mode",
                             disabled=repeated,
                             help="反復測定 (同じ個体の複数時点) がある場合や、共変量を考慮すべき場合は、単純な置換は使えません。"
                                  "condition 内で置換すると、群間差を保ったまま群内の関連を検定します")
    n_perm = c2.selectbox("置換の回数", [199, 499, 999, 1999], index=2, key="cca_perm", disabled=perm_mode == "行わない")
    do_cv = c3.checkbox("交差検証 (個体単位の一つ抜き) を行う", value=False, key="cca_cv",
                        help="補完・標準化・重みの推定を訓練データ内で行い、除いた個体の正準変量を予測して相関を取ります")
    if repeated:
        st.caption("同じ個体の複数の行 (反復測定) があるため、単純な置換検定は行いません。")
    strata = tuple(sub[COND_COL].fillna("").values)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        core, diag, pvals, loo, stab, cv = _cca_all(
            X.values, Y.values, tuple(units), reg, perm_mode if not repeated else "行わない", int(n_perm), strata,
            tuple(X.columns), tuple(Y.columns), do_cv, X0.values, Y0.values, log_x, log_y)
    diag_rows.update({k: v for k, v in diag.items() if k not in diag_rows})
    st.dataframe(pd.DataFrame({"項目": list(diag_rows), "値": [f"{v:.3g}" if isinstance(v, float) else str(v)
                                                             for v in diag_rows.values()]}), hide_index=True)
    if warns:
        st.caption("数値計算の警告: " + "; ".join(sorted({str(w.message)[:80] for w in warns})))

    k = len(core["r"])
    lx = pd.DataFrame(core["load_x"], index=X.columns, columns=[f"CV{i + 1}" for i in range(k)])
    ly = pd.DataFrame(core["load_y"], index=Y.columns, columns=[f"CV{i + 1}" for i in range(k)])

    t_corr, t_cca, t_mode, t_stab = st.tabs(["個別相関", "CCA の結果 (負荷量)", "同じモードに寄与する組み合わせ", "安定性"])
    with t_corr:
        _pairwise_section(sub, X0, Y0, conds, pal, simulated)
    with t_cca:
        tbl = pd.DataFrame({"成分": [f"CV{i + 1}" for i in range(k)], "正準相関 (学習データ内)": core["r"]})
        if pvals is not None:
            tbl[f"置換検定 p ({perm_mode})"] = pvals[:k]
        if cv is not None:
            st.markdown(f"交差検証 (個体単位の一つ抜き, 第 1 成分) の相関: **{cv:.3f}** (未知試料への予測の目安)")
        st.dataframe(tbl.head(10).round(4), hide_index=True)
        st.caption("正準相関は学習データ内の値で、小標本では高くなりやすい値です。p 値・交差検証は任意の参考情報です。")
        _loadings_section(core, lx, ly, gx, gy, pal, simulated, sub, conds)
    with t_mode:
        st.caption("同じ正準モード (CV1) に寄与する X と Y の変数の組み合わせです。探索スコア = X 側の負荷量 x Y 側の負荷量 "
                   "(どちらも第 1 正準モードとの相関)。**個別の相関係数や p 値ではありません**。"
                   "個別に相関が高いペアは「個別相関」タブを見てください。")
        pairs = pd.DataFrame([{"X": a, "Y": b, "X 負荷量 (CV1)": lx.at[a, "CV1"], "Y 負荷量 (CV1)": ly.at[b, "CV1"],
                               "探索スコア (負荷量の積)": lx.at[a, "CV1"] * ly.at[b, "CV1"]}
                              for a in lx.index for b in ly.index])
        pairs = pairs.reindex(pairs["探索スコア (負荷量の積)"].abs().sort_values(ascending=False).index)
        if simulated:
            pairs["注記"] = "模擬データを含む"
        st.dataframe(pairs.head(30).round(3), hide_index=True, height=360)
        st.download_button("組み合わせ (CSV)", to_csv_bytes(pairs), "cca_mode_pairs.csv", "text/csv", key="_dl_cca_pairs")
    with t_stab:
        st.caption("生物学的単位 (個体) を 1 つずつ除いて再解析し、第 1 モードの負荷量と上位の変数がどれだけ変わるかを示します。"
                   "これは探索結果の安定性であり、未知試料への予測性能ではありません。符号は全データの負荷量に合わせています。")
        if loo.empty:
            st.info("除外再解析を行えませんでした (単位が少なすぎます)")
        else:
            unclear = int((loo.get("CV1 対応", pd.Series(dtype=str)) == "不明確").sum())
            st.markdown(f"除外再解析 {len(loo)} 回のうち、第 1 モードの対応が不明確 (負荷量の一致 |r| < 0.7) だった回: **{unclear}**")
            st.dataframe(stab.round(3), hide_index=True, height=320)
            with st.expander("除外再解析ごとの結果"):
                st.dataframe(loo.round(3), hide_index=True)
            st.download_button("安定性 (CSV)", to_csv_bytes(stab), "cca_stability.csv", "text/csv", key="_dl_cca_stab")

    # 出力: 正準係数・スコア
    with st.expander("正準係数とスコア (出力用)"):
        st.caption("正準係数は線形合成を作る重みです。変数間の相関 (多重共線性) で不安定になりやすく、"
                   "係数の大きさで変数を順位付けしないでください。解釈には負荷量を使ってください。")
        coef = pd.concat([pd.DataFrame(core["A"], index=X.columns).assign(側=gx),
                          pd.DataFrame(core["B"], index=Y.columns).assign(側=gy)])
        coef.columns = [f"CV{i + 1}" if isinstance(i, int) else i for i in coef.columns]
        st.dataframe(coef.round(4), height=240)
        scores = pd.DataFrame({LABEL_COL: sub[LABEL_COL].values, COND_COL: sub[COND_COL].values,
                               **{f"{gx} CV{i + 1}": core["U"][:, i] for i in range(min(k, 3))},
                               **{f"{gy} CV{i + 1}": core["V"][:, i] for i in range(min(k, 3))}})
        st.dataframe(scores.round(4), hide_index=True, height=240)
        c1, c2, c3 = st.columns(3)
        c1.download_button("正準係数 (CSV)", to_csv_bytes(coef, index=True), "cca_coefficients.csv", "text/csv", key="_dl_cca_coef")
        c2.download_button("正準スコア (CSV)", to_csv_bytes(scores), "cca_scores.csv", "text/csv", key="_dl_cca_scores")
        load = pd.concat([lx.assign(側=gx, 種類="負荷量"), ly.assign(側=gy, 種類="負荷量"),
                          pd.DataFrame(core["cross_x"], index=X.columns, columns=lx.columns).assign(側=gx, 種類="交差負荷量"),
                          pd.DataFrame(core["cross_y"], index=Y.columns, columns=ly.columns).assign(側=gy, 種類="交差負荷量")])
        c3.download_button("負荷量・交差負荷量 (CSV)", to_csv_bytes(load, index=True), "cca_loadings.csv", "text/csv",
                           key="_dl_cca")
        note = EXPLORE_NOTE + (" ・ 模擬データを含む" if simulated else "")
        register_output("CCA/負荷量・交差負荷量", load.assign(注記=note))
        register_output("CCA/正準係数", coef.assign(注記=note))
        register_output("CCA/正準スコア", scores.assign(注記=note))
        register_output("CCA/正準相関", tbl.assign(注記=note))
        register_output("CCA/安定性 (除外再解析)", stab.assign(注記=note) if len(stab) else stab)


def _pairwise_section(sub, X0, Y0, conds, pal, simulated):
    st.caption("X と Y の各ペアの相関 (欠測はペアごとに除外)。**生データの相関で、群や共変量は考慮していません**。"
               "群の違いによる見かけの相関を避けたい場合は、使う condition を 1 つに絞ってください。")
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("相関係数", ["Spearman", "Pearson"], key="cca_cor_method")
    corr = c2.selectbox("多重性補正", list(ls.CORRECTIONS), key="cca_cor_corr")
    with c3:
        cmap_name = colormap_select("cca_cmap")
    tbl = ls.pairwise_correlation_table(X0, Y0, method.lower(), ls.CORRECTIONS[corr] or "fdr_bh")
    if corr == "補正なし":
        tbl["補正後 p"] = tbl["p"]
    R = tbl.pivot(index="X", columns="Y", values="r").reindex(index=X0.columns, columns=Y0.columns)
    N = tbl.pivot(index="X", columns="Y", values="有効 n").reindex(index=X0.columns, columns=Y0.columns)
    Q = tbl.pivot(index="X", columns="Y", values="補正後 p").reindex(index=X0.columns, columns=Y0.columns)
    fig = go.Figure(go.Heatmap(z=R.values, x=R.columns, y=R.index, zmin=-1, zmax=1, zmid=0, xgap=1, ygap=1,
                               colorscale=resolve_colormap(cmap_name, scale(pal["div"])), colorbar=dict(title="r"),
                               customdata=np.dstack([N.values, Q.values]),
                               hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<br>有効 n = %{customdata[0]}"
                                             "<br>補正後 p = %{customdata[1]:.3g}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", tickfont_size=9)
    fig.update_xaxes(tickangle=-90, tickfont_size=9)
    fig.update_layout(height=max(420, 14 * len(R) + 220), margin=dict(l=200, b=160, t=60))
    show_chart(st, _mark(fig, simulated))
    st.dataframe(tbl.sort_values("p").round(4), hide_index=True, height=260)
    st.caption("小標本では補正後 p が有意にならなくても、探索のために相関は表示しています。")
    register_output("CCA/個別相関", tbl.assign(注記=EXPLORE_NOTE + (" ・ 模擬データを含む" if simulated else "")))
    st.download_button("個別相関 (CSV)", to_csv_bytes(tbl), "cca_pairwise.csv", "text/csv", key="_dl_cca_r")
    c1, c2 = st.columns(2)
    a = c1.selectbox("散布図: X の変数", list(X0.columns), key="cca_pair_x")
    b = c2.selectbox("散布図: Y の変数", list(Y0.columns), key="cca_pair_y")
    d = pd.DataFrame({"x": X0[a].values, "y": Y0[b].values, LABEL_COL: sub[LABEL_COL].values,
                      COND_COL: sub[COND_COL].replace("", "未設定").values}).dropna()
    cats = [c for c in conds + ["未設定"] if (d[COND_COL] == c).any()]
    cmap = color_map(cats, pal)
    fig = go.Figure()
    for c in cats:
        g = d[d[COND_COL] == c]
        fig.add_trace(go.Scatter(x=g["x"], y=g["y"], mode="markers+text", name=c, text=g[LABEL_COL],
                                 textposition="top center", textfont=dict(size=10),
                                 marker=dict(size=10, color=cmap[c], line=dict(width=1.5, color="white")),
                                 hovertemplate="%{text}<br>%{x:.4g}, %{y:.4g}<extra>" + c + "</extra>"))
    row = tbl[(tbl["X"] == a) & (tbl["Y"] == b)].iloc[0]
    fig.update_layout(height=480, xaxis_title=a, yaxis_title=b,
                      title=f"{a} vs {b} ・ r = {row['r']:.3f} (有効 n = {row['有効 n']}, 補正後 p = {row['補正後 p']:.3g})")
    show_chart(st, _mark(fig, simulated))


def _loadings_section(core, lx, ly, gx, gy, pal, simulated, sub, conds):
    k = lx.shape[1]
    st.markdown("**負荷量** (各変数と自分の側の正準変量の相関)。解釈はまず負荷量で行ってください。")
    if k >= 2:
        fig = go.Figure()
        t = np.linspace(0, 2 * np.pi, 200)
        for rad, dash in [(1, "solid"), (0.5, "dot")]:
            fig.add_trace(go.Scatter(x=rad * np.cos(t), y=rad * np.sin(t), mode="lines", hoverinfo="skip",
                                     line=dict(color="#c3c2b7", width=1, dash=dash), showlegend=False))
        for name, L, color in [(gx, lx, pal["series"][0]), (gy, ly, pal["series"][1])]:
            fig.add_trace(go.Scatter(x=L["CV1"], y=L["CV2"], mode="markers+text", name=name, text=L.index,
                                     textposition="top center", textfont=dict(size=9), marker=dict(size=8, color=color),
                                     hovertemplate="%{text}<br>CV1 %{x:.3f}<br>CV2 %{y:.3f}<extra>" + name + "</extra>"))
        fig.update_xaxes(range=[-1.1, 1.1], title="正準変量 CV1 との相関")
        fig.update_yaxes(range=[-1.1, 1.1], title="正準変量 CV2 との相関", scaleanchor="x")
        fig.update_layout(height=640, title="相関円 (負荷量)", margin=dict(t=90))
        show_chart(st, _mark(fig, simulated))
    load = pd.concat([
        pd.DataFrame({"側": gx, "変数": lx.index, "負荷量 CV1": lx["CV1"], "交差負荷量 CV1": core["cross_x"][:, 0]}),
        pd.DataFrame({"側": gy, "変数": ly.index, "負荷量 CV1": ly["CV1"], "交差負荷量 CV1": core["cross_y"][:, 0]}),
    ]).sort_values("負荷量 CV1", key=np.abs, ascending=False)
    st.dataframe(load.round(4), hide_index=True, height=300)
    st.caption("交差負荷量 = 変数と相手側の正準変量との相関 (相手のグループとの共同変動をどれだけ反映するか)。")
    st.markdown("**正準スコア** (各試料の正準変量の値)")
    sc = pd.DataFrame({"u": core["U"][:, 0], "v": core["V"][:, 0], LABEL_COL: sub[LABEL_COL].values,
                       COND_COL: sub[COND_COL].replace("", "未設定").values})
    cats = [c for c in conds + ["未設定"] if (sc[COND_COL] == c).any()]
    cmap = color_map(cats, pal)
    fig = go.Figure()
    for c in cats:
        g = sc[sc[COND_COL] == c]
        fig.add_trace(go.Scatter(x=g["u"], y=g["v"], mode="markers+text", name=c, text=g[LABEL_COL],
                                 textposition="top center", textfont=dict(size=10),
                                 marker=dict(size=10, color=cmap[c], line=dict(width=1.5, color="white")),
                                 hovertemplate="%{text}<br>X 側 %{x:.3f}<br>Y 側 %{y:.3f}<extra>" + c + "</extra>"))
    fig.update_layout(height=460, xaxis_title=f"{gx} の正準変量 CV1", yaxis_title=f"{gy} の正準変量 CV1",
                      title=f"CV1 のスコア (学習データ内 r = {core['r'][0]:.3f})", margin=dict(t=90))
    show_chart(st, _mark(fig, simulated))
