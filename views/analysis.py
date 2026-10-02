"""解析ページ: LC-MS/MS 定量結果の整形・QC・可視化・統計解析 (app.py から呼ばれる)。"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

import lcms_analysis as la
import lcms_qc as qc
import lcms_stats as ls
from lcms_analysis import (
    CONC, COND_COL, CPD_COL, DATASET_COL, FC, GROUP_COL, LOG2FC, MEAN_SD, MEDIAN_IQR, VALUE_COL,
)
from lcms_plots import bar_samples, bar_summary, clustered_heatmap, heatmap_grid, resolve_colormap, scale, scatter_2d
from make_conc_table import FILE_COL, IS_NAME, LABEL_COL, MISSING, is_std
from ui_common import colormap_select, colorscale_for, compound_picker, compound_select, theme, to_csv_bytes
from ui_qc import accuracy_tab, calibration_choices, calibration_tab, drift_tab, merged_table
from ui_extra import limits_tab, multigroup_tab, ratio_definitions, ratio_view
from ui_project import check_data, loader, saver
from ui_stats import correlation_tab, kegg_tab, scatter_tab, volcano_tab

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SINGLE, MULTI = "単一データセット", "複数データセット統合"
BY_GROUP, MERGED = "希釈グループごとに解析", "検量線範囲で統合 (1 サンプル 1 行)"
UNSET = "未設定"


@st.cache_data(show_spinner="読み込み中...")
def load_tables(raw_bytes, is_name):
    return la.load_tables(raw_bytes, is_name)


@st.cache_data(show_spinner=False)
def load_full(raw_bytes):
    return qc.parse_full(raw_bytes)


@st.cache_data(show_spinner="UMAP 計算中...")
def run_umap(values, n_neighbors, min_dist, seed):
    import umap  # 読み込みが重いので遅延 import

    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed).fit_transform(values)


def ref_value(kind):
    return {FC: 1, LOG2FC: 0}.get(kind)


def stat_label(stat):
    return "平均" if stat == MEAN_SD else "中央値"


def range_label(stat):
    return "± SD" if stat == MEAN_SD else "IQR"


# ---------------------------------------------------------------- condition 入力
USE_COL = "使用"
BULK_COND, BULK_OFF, BULK_ON = "condition を設定", "解析に使わない", "解析に使う"


def condition_editor(ds, labels):
    """label ごとに condition と 解析に使うか を入力する表。({label: condition}, 使う label の集合) を返す。"""
    ss = st.session_state
    init_key, ver_key, cur_key = f"cond_init::{ds}", f"cond_ver::{ds}", f"cond_cur::{ds}"
    if init_key not in ss or list(ss[init_key][LABEL_COL]) != list(labels):
        # label の一覧が変わったとき (表記ゆれの統一など) は、入力済みの内容を label で引き継ぐ
        prev = ss.get(cur_key, ss.get(init_key))
        old = prev.set_index(LABEL_COL) if prev is not None else pd.DataFrame(columns=[COND_COL, USE_COL])
        ss[init_key] = pd.DataFrame({
            LABEL_COL: labels,
            COND_COL: [old[COND_COL].get(lb, "") for lb in labels],
            USE_COL: [bool(old[USE_COL].get(lb, True)) for lb in labels],
        })
        ss[ver_key] = ss.get(ver_key, -1) + 1

    def reset_editor(df):
        ss[init_key] = df
        ss[ver_key] += 1

    with st.expander("一括入力 / 対応表 CSV の読み込み"):
        c1, c2, c3, c4 = st.columns([2, 1.3, 2, 1])
        pat = c1.text_input("label に含まれる文字列 (正規表現可)", key=f"pat::{ds}", placeholder="例: ^D-(1|2|3|4|5)$")
        action = c2.selectbox("操作", [BULK_COND, BULK_OFF, BULK_ON], key=f"act::{ds}")
        val = c3.text_input("設定する condition", key=f"val::{ds}", placeholder="例: Control",
                            disabled=action != BULK_COND)
        if c4.button("一括設定", key=f"_btn_bulk::{ds}", disabled=not pat):
            cur = ss.get(cur_key, ss[init_key]).copy()
            try:
                with warnings.catch_warnings():  # 捕捉グループを含む正規表現の UserWarning を抑制
                    warnings.simplefilter("ignore", UserWarning)
                    mask = cur[LABEL_COL].str.contains(pat, regex=True)
            except Exception as e:
                st.error(f"正規表現が不正です: {e}")
            else:
                if action == BULK_COND:
                    cur.loc[mask, COND_COL] = val
                else:
                    cur.loc[mask, USE_COL] = action == BULK_ON
                reset_editor(cur)
                st.rerun()
        up = st.file_uploader("対応表 CSV (列: label, condition, 使用 [任意])", type=["csv"], key=f"_up_map::{ds}")
        if up is not None and ss.get(f"map_done::{ds}") != up.file_id:
            try:
                m = pd.read_csv(up, dtype=str, encoding="utf-8-sig").fillna("")
                mapping = dict(zip(m[LABEL_COL].str.strip(), m[COND_COL].str.strip()))
            except Exception as e:
                st.error(f"CSV を読めません (label, condition 列が必要です): {e}")
            else:
                cur = ss.get(cur_key, ss[init_key]).copy()
                cur[COND_COL] = [mapping.get(lb, c) for lb, c in zip(cur[LABEL_COL], cur[COND_COL])]
                if USE_COL in m:
                    use = dict(zip(m[LABEL_COL].str.strip(), m[USE_COL].str.strip().str.lower()))
                    cur[USE_COL] = [use.get(lb, str(u)).lower() in ("true", "1", "yes", "○", "使用")
                                    for lb, u in zip(cur[LABEL_COL], cur[USE_COL])]
                ss[f"map_done::{ds}"] = up.file_id
                reset_editor(cur)
                st.rerun()

    edited = st.data_editor(
        ss[init_key], key=f"cond_editor::{ds}::{ss[ver_key]}", hide_index=True, num_rows="fixed",
        disabled=[LABEL_COL], height=min(600, 36 * (len(labels) + 1)),
        column_config={
            COND_COL: st.column_config.TextColumn(COND_COL, help="実験群名 (空欄 = 条件ごとの集計から除外)"),
            USE_COL: st.column_config.CheckboxColumn(USE_COL, help="チェックを外したサンプルは、QC 以外のすべての解析から除外します"),
        },
    )
    ss[cur_key] = edited
    cond_map = {lb: str(c).strip() for lb, c in zip(edited[LABEL_COL], edited[COND_COL]) if str(c).strip()}
    used = set(edited.loc[edited[USE_COL].fillna(True).astype(bool), LABEL_COL])
    n_off = len(labels) - len(used)
    if n_off:
        st.caption(f"解析に使わない label: {n_off} 件 ({', '.join(sorted(set(labels) - used, key=la.natural_key))})")
    st.download_button("対応表 CSV をダウンロード", to_csv_bytes(edited), f"{ds}_condition.csv", "text/csv",
                       key=f"_dl_map::{ds}")
    return cond_map, used


def control_select(ds, conds):
    if not conds:
        st.info("condition を入力すると対照群を選べます")
        return None
    return st.selectbox("対照群 (Fold Change の分母)", conds, key=f"control::{ds}")


# ---------------------------------------------------------------- 表示オプション
def view_controls(key, has_control, kinds):
    """表示単位・値の種類・統計量を選ぶ。"""
    c1, c2, c3 = st.columns(3)
    unit = c1.radio("表示単位", ["サンプルごと", "condition ごと (要約)"], key=f"{key}_unit")
    kinds = kinds if has_control else [k for k in kinds if k not in (FC, LOG2FC)]
    kind = c2.radio("値", kinds, key=f"{key}_kind",
                    help=None if has_control else "Fold Change は条件設定タブで対照群を選ぶと使えます")
    stat = c3.radio("統計量", [MEAN_SD, MEDIAN_IQR], key=f"{key}_stat", disabled=unit == "サンプルごと")
    return unit.startswith("condition"), kind, stat


def summary_hover(summ, stat, index_col):
    """ヒートマップのツールチップ用に 範囲 と n を文字列にする。"""
    txt = summ.assign(h=[
        f"<br>{range_label(stat)}: {lo:.3g} – {hi:.3g}<br>n = {int(n)}" if n else ""
        for lo, hi, n in zip(summ["lower"], summ["upper"], summ["n"])
    ])
    return txt.pivot(index=index_col, columns=CPD_COL, values="h")


# ================================================================ サイドバー
st.title("LC-MS/MS 定量解析")
pal = theme()

with st.sidebar:
    loader(DATA_DIR)  # 設定の読み込みは、ほかのウィジェットより先に行う必要がある
    mode = st.radio("モード", [SINGLE, MULTI], horizontal=True, key="mode")
    st.header("1. データ")
    local = sorted(DATA_DIR.glob("*.txt"))
    sources = []
    if mode == SINGLE:
        up = st.file_uploader("LabSolutions の定量結果 (.txt)", type=["txt"], key="_up_data")
        if up is not None:
            sources = [(Path(up.name).stem, up.getvalue())]
        elif local:
            pick = st.selectbox("または data/ から選択", local, format_func=lambda p: p.name, key="data_pick")
            sources = [(pick.stem, pick.read_bytes())]
    else:
        ups = st.file_uploader("LabSolutions の定量結果 (.txt, 複数可)", type=["txt"], accept_multiple_files=True,
                               key="_up_data_multi")
        sources = [(Path(u.name).stem, u.getvalue()) for u in ups]
        picks = st.multiselect("data/ から追加", local, format_func=lambda p: p.name, key="data_picks")
        sources += [(p.stem, p.read_bytes()) for p in picks]
    if not sources:
        st.info("データファイルを選んでください")
        st.stop()
    names = la.unique_names([n for n, _ in sources])  # 同名ファイルは #2 を付けて区別
    sources = [(n, b) for n, (_, b) in zip(names, sources)]

    try:
        loaded = [load_tables(b, IS_NAME) for _, b in sources]
    except ValueError as e:
        st.error(str(e))
        st.stop()

    def is_candidates(raw, cpds):
        """内部標準の候補: STD 以外の全サンプルで検出された化合物 (欠けていると補正できないサンプルが出るため)。"""
        samples = raw[~raw[LABEL_COL].map(is_std)]
        return [c for c in cpds if len(samples) and samples[c].notna().all()]

    sets = [is_candidates(raw, cpds) for raw, _, cpds in loaded]
    common = [c for c in sets[0] if all(c in s for s in sets)]
    if not common:
        st.error("全サンプルで検出された化合物が無いため、内部標準を選べません")
        st.stop()
    is_name = st.selectbox("内部標準", common, index=common.index(IS_NAME) if IS_NAME in common else 0, key="is_name",
                           help="全サンプル (STD を除く) で検出された化合物だけを候補にしています")
    na_rep = st.text_input("CSV の欠損値表記", value="", key="na_rep", help=f"未検出 ({MISSING}) を置き換える文字。空欄のままでも可")

    st.header("2. 解析に使うデータ")
    source = st.radio("濃度", ["IS 補正後", "補正前 (生データ)"], key="source")
    include_std = source.startswith("補正前") and st.checkbox("STD を含める", value=False, key="include_std")
    dilution_mode = st.radio("希釈の扱い", [BY_GROUP, MERGED], key="dilution_mode",
                             help="「検量線範囲で統合」は 検量線・希釈 タブの設定で希釈なし / 希釈測定を化合物ごとに選び、"
                                  "1 サンプル 1 行にまとめます (希釈測定は希釈倍率を掛けた値)")
    merge_mode = dilution_mode == MERGED
    mult_dilution = st.checkbox("希釈倍率を掛けて元濃度に換算", value=False, disabled=merge_mode, key="mult_dilution",
                                help="x10 のサンプルの値を 10 倍して表示・解析します")
    exclude_is = st.checkbox("内部標準を化合物から除外", value=True, key="exclude_is")
    harmonize = st.checkbox("label の「_」を「-」に統一", value=False, key="harmonize",
                            help="例: D_1 と D-1 を同じサンプルとして扱います (ファイル名の表記ゆれ対策)")

    st.header("定量限界・検出限界")
    mask_level = st.selectbox("マスク", [qc.MASK_NONE, qc.MASK_LOD, qc.MASK_LOQ], key="mask_level",
                              help="測定ごとの LOD / LOQ (LabSolutions が S/N から算出) を下回る値を扱いを変えます。"
                                   "判定は IS 補正前の算出濃度で行います (S/N = ∞ で限界値が 0 の測定は対象外)")
    mask_repl = st.radio("マスクした値", [qc.REPL_NAN, qc.REPL_HALF, qc.REPL_LIM], key="mask_repl",
                         disabled=mask_level == qc.MASK_NONE)

    st.header("3. 正規化")
    norm_method = st.selectbox("サンプル間の正規化", ls.NORMALIZATIONS, key="norm_method",
                               help="内部標準による補正の後に、サンプルごとの希釈の違いを補正します。詳しくは「解析手法の解説」ページ")
    norm_ref = st.radio("PQN の参照", [ls.REF_ALL, ls.REF_CONTROL], disabled=norm_method != ls.NORM_PQN, key="norm_ref")
    pqn_total = st.checkbox("PQN の前に総量正規化 (Dieterle et al., 2006 の手順)", value=True, key="pqn_total",
                            disabled=norm_method != ls.NORM_PQN)

def fix_labels(df):
    if not harmonize or df is None:
        return df
    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].str.replace("_", "-", regex=False)
    return df


check_data(sources)
datasets = []
for name, b in sources:
    raw_df, is_df, cpds = load_tables(b, is_name)
    raw_df, is_df, full = fix_labels(raw_df), fix_labels(is_df), fix_labels(load_full(b))
    df = is_df if source == "IS 補正後" else raw_df
    df, n_masked = qc.apply_limit_mask(df, full, cpds, mask_level, mask_repl, skip=(is_name,))
    mask_info = (f"{name}: {mask_level} の値 {n_masked} 個を「{mask_repl}」で処理しています (内部標準を除く)。"
                 if mask_level != qc.MASK_NONE else "")
    datasets.append(dict(name=name, raw=raw_df, is_df=is_df, compounds=cpds, full=full, df=df, mask_info=mask_info))


def detected(base, cpds):
    return [c for c in cpds if base[c].notna().any() and not (exclude_is and c == is_name)]


def make_base(d, cond_map, control, merged=None, used=None, cond_filter=None):
    """解析用の表を作る: (検量線範囲での統合) -> condition 付与 -> サンプルの選択 -> 正規化。

    used: 解析に使う label の集合 (None = すべて)、cond_filter: 残す condition (空 = すべて)。
    サンプルの選択は正規化の前に行う (除外したサンプルが PQN の参照などに影響しないように)。
    (表, 化合物, 正規化係数) を返す。
    """
    df, inc_std, mult = d["df"], include_std, mult_dilution
    if merge_mode:
        if merged is None:
            ranges, choices = calibration_choices(d["name"], d["full"], d["raw"], d["compounds"], editable=False)
            merged, _, _ = merged_table(d["df"], d["raw"], d["compounds"], ranges, choices)
        df, inc_std, mult = merged, False, False
    base = la.build_base(df, d["compounds"], cond_map, inc_std, mult)
    keep = pd.Series(True, index=base.index)
    if used is not None:
        keep &= base[LABEL_COL].isin(used) | base[LABEL_COL].map(is_std)
    if cond_filter:
        keep &= base[COND_COL].isin(cond_filter) | base[LABEL_COL].map(is_std)
    base = base[keep]
    cpds = detected(base, d["compounds"])
    try:
        base, factors = ls.normalize(base, cpds, norm_method, norm_ref, control, pqn_total)
    except ValueError as e:
        st.warning(f"{d['name']}: 正規化できません ({e})。正規化なしで続けます。")
        factors = pd.DataFrame()
    return base, cpds, factors


# ================================================================ 単一データセット
def single_mode(d):
    tabs = st.tabs(["条件設定", "検量線・希釈", "STD 正確さ", "LOD・LOQ", "ドリフト", "テーブル", "代謝物比", "棒グラフ",
                    "ヒートマップ", "散布図", "クラスタリング", "PCA", "UMAP", "ボルケーノ", "多群比較", "相関", "KEGG"])
    (tab_cond, tab_cal, tab_acc, tab_lim, tab_drift, tab_tbl, tab_ratio, tab_bar, tab_hm, tab_sc,
     tab_cl, tab_pca, tab_umap, tab_vol, tab_mg, tab_cor, tab_kegg) = tabs
    all_labels = sorted(d["raw"][LABEL_COL].unique(), key=la.natural_key)

    with tab_cond:
        st.caption("label ごとに実験群 (condition) を入力し、解析に使うサンプルを選んでください (「使用」のチェック)。"
                   "x10 と希釈なしの同じ label には同じ設定が付きます。Excel などからのコピー & ペーストもできます。")
        c1, c2 = st.columns([3, 1])
        with c1:
            cond_map, used = condition_editor(d["name"], all_labels)
        conds = la.condition_order(cond_map, all_labels)
        with c2:
            control = control_select(d["name"], conds)
            conds = la.condition_order(cond_map, all_labels, control)
            if conds:
                cnt = pd.Series(cond_map).value_counts().reindex(conds)
                st.dataframe(cnt.rename("label 数"), height=min(400, 36 * (len(conds) + 1)))

    with tab_cal:
        merged = calibration_tab(d["name"], d["full"], d["df"], d["raw"], d["compounds"], na_rep)
    with tab_acc:
        accuracy_tab(d["full"], pal)
    with tab_lim:
        limits_tab(d["full"], pal, d["mask_info"])
    with tab_drift:
        drift_tab(d["full"], is_name, pal, cond_map)
    with tab_ratio:
        ratios = ratio_definitions(d["compounds"])
        add_ratio = st.checkbox("比を他の解析にも追加 (棒グラフ・ヒートマップ・散布図・検定など)", value=True, key="ratio_add")

    with st.sidebar:
        st.header("4. 解析に使うサンプル")
        cond_filter = st.multiselect("condition で絞り込み (空欄 = すべて)", conds, key=f"cond_filter::{d['name']}",
                                     help="個別のサンプルの除外は 条件設定 タブの「使用」列で行います")
    base, compounds, factors = make_base(d, cond_map, control, merged, used, cond_filter)
    if ratios:
        base = la.add_ratios(base, ratios)  # 比は正規化係数に依存しないので、正規化の後に計算する
        if add_ratio:
            compounds = compounds + [r[0] for r in ratios]
    n_all = int((~d["raw"].drop_duplicates(LABEL_COL)[LABEL_COL].map(is_std)).sum())
    n_used = base.loc[~base[LABEL_COL].map(is_std), LABEL_COL].nunique()
    if n_used < n_all:
        st.sidebar.caption(f"解析に使う label: {n_used} / {n_all}")
    groups = la.group_order(base[GROUP_COL])
    has_control = control is not None and (base[COND_COL] == control).any()
    st.caption(
        f"{d['name']} ・ 解析対象 {len(base)} サンプル ・ 検出化合物 {len(compounds)} / {len(d['compounds'])} ・ "
        f"希釈グループ: {', '.join(f'{g} (n={(base[GROUP_COL] == g).sum()})' for g in groups)} ・ "
        f"condition: {len(conds)} 群" + (f" (対照群: {control})" if has_control else "")
        + f" ・ 希釈: {dilution_mode} ・ 正規化: {norm_method}"
    )

    # ------------------------------------------------ テーブル
    with tab_tbl:
        st.subheader("濃度表 (補正前)")
        raw_c = la.with_condition(d["raw"], cond_map)
        st.dataframe(raw_c, hide_index=True, height=320)
        st.download_button("CSV をダウンロード", to_csv_bytes(raw_c, na_rep), f"{d['name']}_conc.csv", "text/csv",
                           key="_dl_raw")

        st.subheader(f"IS 補正後の換算濃度 (内部標準: {is_name}, STD を除く)")
        if d["is_df"] is None:
            st.warning("内部標準がデータにありません")
        else:
            samples = d["raw"][~d["raw"][LABEL_COL].map(is_std)]
            means = samples.groupby(samples[la.DILUTION_COL].map(la.group_name))[is_name].agg(["mean", "count"])
            st.markdown(
                "IS 平均: " + " ・ ".join(f"**{g}** {r['mean']:.3f} (n={int(r['count'])})" for g, r in means.iterrows())
                + "  \n比率 = IS 濃度 / 希釈グループ内平均、換算濃度 = 濃度 × (1 / 比率)"
            )
            is_c = la.with_condition(d["is_df"], cond_map)
            st.dataframe(is_c, hide_index=True, height=320)
            st.download_button("CSV をダウンロード", to_csv_bytes(is_c, na_rep), f"{d['name']}_conc_IS.csv",
                               "text/csv", key="_dl_is")

        st.subheader("解析に使う表")
        st.caption(f"濃度: {source} ・ 希釈: {dilution_mode} ・ 正規化: {norm_method}。棒グラフ以降のタブはこの表を使います。")
        used_cols = [FILE_COL, LABEL_COL, COND_COL, GROUP_COL] + compounds
        st.dataframe(base[used_cols], hide_index=True, height=320)
        st.download_button("CSV をダウンロード", to_csv_bytes(base[used_cols], na_rep), f"{d['name']}_analysis.csv",
                           "text/csv", key="_dl_base")
        if not factors.empty:
            st.markdown("**正規化係数** (値をこの係数で割っています。1 から大きく離れたサンプルは要確認)")
            st.dataframe(factors.round(4), hide_index=True, height=240)

    # ------------------------------------------------ 棒グラフ
    with tab_bar:
        by_cond, kind, stat = view_controls("bar", has_control, [CONC, FC, LOG2FC])
        c1, c2 = st.columns([4, 1])
        with c1:
            sel = compound_select("bar_cpds", compounds, default=compounds[:1], max_selections=40)
        show_points = c2.checkbox("各サンプルの点を重ねる", value=True, disabled=not by_cond, key="bar_pts")
        if sel:
            ytitle = kind
            if by_cond:
                df = base[base[COND_COL] != ""]
                if df.empty:
                    st.warning("condition が入力されていません (条件設定タブ)")
                else:
                    summ, pts = la.summary_values(df, sel, kind, control, stat, [GROUP_COL, COND_COL])
                    st.plotly_chart(bar_summary(summ, pts, COND_COL, conds, GROUP_COL, groups, pal,
                                                f"{ytitle} ({stat_label(stat)})", stat_label(stat), show_points,
                                                ref_value(kind)), width="stretch")
                    st.caption(caption_for(kind, stat))
                    with st.expander("表で見る"):
                        st.dataframe(summ, hide_index=True)
            else:
                vals = la.transform_values(base, sel, kind, control)
                vals = la.order_samples(vals, conds)
                long = vals.melt(id_vars=[FILE_COL, LABEL_COL, GROUP_COL, COND_COL], value_vars=sel,
                                 var_name=CPD_COL, value_name=VALUE_COL)
                labels = list(dict.fromkeys(vals[LABEL_COL]))
                st.plotly_chart(bar_samples(long, labels, GROUP_COL, groups, pal, ytitle, ref_value(kind)),
                                width="stretch")
                st.caption("未検出 (-----) は棒なしで表示しています。"
                           + (" Fold Change の分母は同じ希釈グループの対照群平均です。" if kind != CONC else ""))
                with st.expander("表で見る"):
                    st.dataframe(long.pivot_table(index=LABEL_COL, columns=[CPD_COL, GROUP_COL], values=VALUE_COL)
                                 .reindex(labels), height=400)

    # ------------------------------------------------ ヒートマップ
    with tab_hm:
        by_cond, kind, stat = view_controls(
            "hm", has_control, [CONC, "log10", "Z スコア (化合物ごと)", FC, LOG2FC])
        hm_cpds = compound_picker("hm", compounds)
        c1, c2, c3 = st.columns([4, 1, 1])
        with c2:
            cmap = colormap_select("hm_cmap")
        limit = c3.number_input("色の上限 |log2FC| (0 = 自動)", 0.0, 20.0, 0.0, 0.5, key="hm_lim",
                                disabled=kind != LOG2FC)
        mats, hovers = {}, {}
        lin_kind = kind if kind in (FC, LOG2FC) else CONC
        for g in groups:
            sub = base[base[GROUP_COL] == g]
            if by_cond:
                sub = sub[sub[COND_COL] != ""]
                if sub.empty:
                    continue
                summ, _ = la.summary_values(sub, hm_cpds, lin_kind, control, stat, [COND_COL])
                X = summ.pivot(index=COND_COL, columns=CPD_COL, values="center").reindex(conds).dropna(how="all")
                hovers[g] = summary_hover(summ, stat, COND_COL).reindex(X.index)
            else:
                sub = la.order_samples(la.transform_values(sub, hm_cpds, lin_kind, control), conds)
                X = sub[hm_cpds].astype(float)
                X.index = [f"{lb} [{c}]" if c else lb for lb, c in zip(la.unique_names(sub[LABEL_COL]), sub[COND_COL])]
            X = X.loc[:, X.notna().any()]
            if kind == "log10":
                X = la.log10_safe(X)
            elif kind.startswith("Z"):
                X = la.zscore_cols(X)
            mats[g] = X
        if not mats:
            st.warning("condition が入力されていません (条件設定タブ)")
        else:
            vals = np.concatenate([m.values.ravel() for m in mats.values()])
            cs, title, mid, zr = colorscale_for(kind, pal, vals, limit, cmap)
            if by_cond:
                title = f"{title}<br>({stat_label(stat)})"
            st.plotly_chart(heatmap_grid(mats, cs, title, mid, zr, hovers or None), width="stretch")
            st.caption("空白セルは未検出 (または対照群で未検出)。Z スコアは希釈グループごとに化合物単位で計算しています。"
                       + (" " + caption_for(kind, stat) if by_cond else ""))

    # ------------------------------------------------ 多変量解析
    def group_matrices(opts):
        out = {}
        for g in groups:
            sub = la.order_samples(base[base[GROUP_COL] == g], conds)
            X = la.prep_matrix(sub, opts["compounds"], opts["max_missing"], opts["impute"], opts["log"], False)
            X = ls.scale_columns(X, opts["scaling"])
            if X.shape[0] < 3 or X.shape[1] < 2:
                st.warning(f"{g}: サンプルまたは化合物が少なすぎるため解析できません ({X.shape[0]} x {X.shape[1]})")
                continue
            cond = pd.Series(sub[COND_COL].replace("", UNSET).values, index=la.unique_names(sub[LABEL_COL]))
            out[g] = (X, cond.reindex(X.index))
        return out

    color_groups = conds + [UNSET]

    with tab_sc:
        scatter_tab(base, groups, conds, compounds, pal)

    with tab_cl:
        opts = prep_controls("cl", compounds)
        c1, c2, c3, c4 = st.columns(4)
        method = c1.selectbox("連結法", ["average", "complete", "ward", "single", "weighted", "centroid", "median"], key="cl_method",
                              help="ward / centroid / median はユークリッド距離でのみ使えます")
        euclid_only = method in la.EUCLIDEAN_ONLY
        dist_label = c2.selectbox("距離", list(la.DISTANCES), disabled=euclid_only, key="cl_dist")
        metric = "euclidean" if euclid_only else la.DISTANCES[dist_label]
        p = c3.number_input("ミンコフスキーの p", 1.0, 20.0, 3.0, 0.5, disabled=metric != "minkowski", key="cl_p",
                            help="p = 1 でマンハッタン、p = 2 でユークリッドと同じ")
        with c4:
            cmap = colormap_select("cl_cmap")
        if metric == "braycurtis" and (opts["scaling"] != "なし" or opts["log"]):
            st.warning("ブレイ・カーティス距離は非負の値 (濃度そのまま) を前提とします。"
                       "前処理の log10 変換・スケーリングをオフにしてください。")
        if metric == "mahalanobis":
            st.caption("マハラノビス距離: サンプル数より変数が多いと標本共分散が特異になるため、"
                       "Ledoit-Wolf の縮小推定による共分散の逆行列を使っています。")
        for g, (X, cond) in group_matrices(opts).items():
            if conds:
                X = X.rename(index=lambda s: f"{s} [{cond[s]}]" if cond[s] != UNSET else s)
            if opts["scaling"] != "なし":
                cs, mid = resolve_colormap(cmap, scale(pal["div"])), 0
            else:
                cs, mid = resolve_colormap(cmap, scale(pal["seq"])), None
            try:
                fig, M = clustered_heatmap(X, method, metric, cs, mid, pal,
                                           f"{g}  ({X.shape[0]} サンプル x {X.shape[1]} 化合物, {method} / "
                                           f"{'ユークリッド' if euclid_only else dist_label})", p=p)
            except ValueError as e:
                st.error(f"{g}: {e}")
                continue
            st.plotly_chart(fig, width="stretch")
            st.download_button(f"{g} の並べ替え後の行列 (CSV)", to_csv_bytes(M, index=True),
                               f"cluster_{g}.csv", "text/csv", key=f"_dl_cl_{g}")

    with tab_pca:
        opts = prep_controls("pca", compounds)
        c1, c2, c3 = st.columns(3)
        pcx = c1.number_input("横軸 PC", 1, 10, 1, key="pca_x")
        pcy = c2.number_input("縦軸 PC", 1, 10, 2, key="pca_y")
        show_text = c3.checkbox("ラベルを表示", value=True, key="pca_txt")
        mats = group_matrices(opts)
        cols = st.columns(max(1, len(mats)))
        for i, (g, (X, cond)) in enumerate(mats.items()):
            n = min(X.shape)
            pca = PCA(n_components=n).fit(X.values)
            scores = pd.DataFrame(pca.transform(X.values), index=X.index, columns=[f"PC{k + 1}" for k in range(n)])
            ev = pca.explained_variance_ratio_ * 100
            a, b = min(pcx, n), min(pcy, n)
            with cols[i]:
                st.plotly_chart(scatter_2d(
                    scores, f"PC{a}", f"PC{b}", show_text, f"{g}  ({X.shape[0]} x {X.shape[1]})",
                    f"PC{a} ({ev[a - 1]:.1f}%)", f"PC{b} ({ev[b - 1]:.1f}%)", pal,
                    cond if conds else None, color_groups,
                ), width="stretch", key=f"pca_{g}")
                ev_fig = go.Figure(go.Bar(x=scores.columns[:10], y=ev[:10], marker_color=pal["series"][0],
                                          hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
                ev_fig.update_layout(title="寄与率 (%)", height=240, margin=dict(t=40, b=30), barcornerradius=4)
                st.plotly_chart(ev_fig, width="stretch", key=f"ev_{g}")
                pcs = list(dict.fromkeys([a, b]))  # 横軸と縦軸に同じ PC を選んだ場合は 1 列だけ
                load = pd.DataFrame(pca.components_[[k - 1 for k in pcs]].T, index=X.columns,
                                    columns=[f"PC{k}" for k in pcs])
                with st.expander("ローディング (寄与の大きい化合物)"):
                    st.dataframe(load.reindex(load[f"PC{a}"].abs().sort_values(ascending=False).index).round(3),
                                 height=300)

    with tab_umap:
        opts = prep_controls("umap", compounds)
        c1, c2, c3, c4 = st.columns(4)
        n_neighbors = c1.slider("n_neighbors", 2, 30, 5, key="umap_nn")
        min_dist = c2.slider("min_dist", 0.0, 1.0, 0.1, 0.05, key="umap_md")
        seed = c3.number_input("random_state", 0, 9999, 42, key="umap_seed")
        show_text = c4.checkbox("ラベルを表示", value=True, key="umap_txt")
        mats = group_matrices(opts)
        cols = st.columns(max(1, len(mats)))
        for i, (g, (X, cond)) in enumerate(mats.items()):
            nn = min(n_neighbors, X.shape[0] - 1)
            emb = pd.DataFrame(run_umap(X.values, nn, min_dist, int(seed)), index=X.index, columns=["UMAP1", "UMAP2"])
            with cols[i]:
                fig = scatter_2d(emb, "UMAP1", "UMAP2", show_text, f"{g}  ({X.shape[0]} x {X.shape[1]}, n_neighbors={nn})",
                                 "UMAP1", "UMAP2", pal, cond if conds else None, color_groups)
                fig.update_xaxes(zeroline=False).update_yaxes(zeroline=False)  # UMAP 座標の 0 に意味はない
                st.plotly_chart(fig, width="stretch", key=f"umap_{g}")
        st.caption("サンプル数が少ないため、UMAP の配置は n_neighbors や random_state で大きく変わります。")

    with tab_ratio:
        ratio_view(base, ratios, groups, conds, pal, na_rep)
    with tab_vol:
        vres = volcano_tab(base, groups, conds, control if has_control else None, compounds, pal)
    with tab_mg:
        multigroup_tab(base, groups, conds, compounds, pal, control if has_control else None)
    with tab_cor:
        correlation_tab(base, groups, conds, compounds, pal, vres)
    with tab_kegg:
        kegg_tab(d["compounds"], vres, pal)


def caption_for(kind, stat):
    rng = "誤差棒 = SD" if stat == MEAN_SD else "誤差棒 = 第1–第3四分位 (IQR)"
    if kind == CONC:
        return f"棒 = {stat_label(stat)}、{rng}。"
    base = f"FC = 各サンプル / 同じ希釈グループの対照群平均。棒 = FC の{stat_label(stat)}、{rng}。"
    if kind == LOG2FC:
        base += " log2(FC) は FC スケールで求めた値を log2 変換しています (下端が 0 以下になる誤差棒は省略)。"
    return base


def prep_controls(key, compounds):
    cpds = compound_picker(key, compounds)
    with st.expander("前処理の設定", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        max_missing = c1.slider("欠損率がこれ以下の化合物を使用", 0.0, 1.0, 0.5, 0.05, key=f"{key}_miss")
        impute = c2.selectbox("欠損値の補完", ["最小値の1/2", "0"], key=f"{key}_imp")
        log = c3.checkbox("log10 変換", value=True, key=f"{key}_log")
        scaling = c4.selectbox("スケーリング (化合物ごと)", ls.SCALINGS, index=1, key=f"{key}_scale",
                               help="van den Berg et al. (2006)。詳しくは「解析手法の解説」ページ")
    return dict(max_missing=max_missing, impute=impute, log=log, scaling=scaling, compounds=cpds)


# ================================================================ 複数データセット統合
def multi_mode(datasets):
    tab_cond, tab_ratio, tab_tbl, tab_bar, tab_hm = st.tabs(
        ["条件設定", "代謝物比", "統合テーブル", "棒グラフ (FC)", "ヒートマップ (FC)"])
    with tab_ratio:
        all_cpds = list(dict.fromkeys(c for d in datasets for c in d["compounds"]))
        ratios = ratio_definitions(all_cpds)
        st.caption("定義した比は、各データセットで計算して Fold Change の統合にも使います。")
    ds_names = [d["name"] for d in datasets]
    if len(ds_names) > len(pal["series"]):
        st.warning(f"データセットが {len(ds_names)} 個あります。色は {len(pal['series'])} 色までなので、"
                   "棒グラフでは色が重複します (ヒートマップはパネルで区別されます)。")

    with tab_cond:
        st.caption("データセットごとに condition と対照群を設定してください。Fold Change は各データセット内で、"
                   "それぞれの対照群の平均を分母に計算します。")
        settings = {}
        for i, d in enumerate(datasets):
            labels = sorted(d["raw"][LABEL_COL].unique(), key=la.natural_key)
            with st.expander(d["name"], expanded=i == 0):
                c1, c2 = st.columns([3, 1])
                with c1:
                    cond_map, used = condition_editor(d["name"], labels)
                with c2:
                    control = control_select(d["name"], la.condition_order(cond_map, labels))
                settings[d["name"]] = (cond_map, la.condition_order(cond_map, labels, control), control, used)

    # 各データセットで FC を計算して縦に連結
    bases, all_groups = {}, set()
    for d in datasets:
        cond_map, conds, control, used = settings[d["name"]]
        b, _, _ = make_base(d, cond_map, control, used=used)
        b = b[b[COND_COL] != ""]
        if control is None or not (b[COND_COL] == control).any():
            continue
        ok = [r for r in ratios if all(t in d["compounds"] for t in r[1] + r[2])]
        b = la.add_ratios(b, ok)
        bases[d["name"]] = (b, conds, control, d["compounds"] + [r[0] for r in ok])
        all_groups |= set(b[GROUP_COL])
    skipped = [n for n in ds_names if n not in bases]
    if skipped:
        st.warning("condition / 対照群が未設定のため統合から除外: " + ", ".join(skipped))
    if not bases:
        st.info("条件設定タブで各データセットの condition と対照群を設定してください。")
        return

    groups = la.group_order(all_groups)
    c1, c2, c3, c4 = st.columns(4)
    group = c1.selectbox("希釈グループ", groups, key="m_group", help="FC は希釈グループ内で計算します。統合する希釈グループを選んでください")
    kind = c2.radio("値", [LOG2FC, FC], key="m_kind")
    stat = c3.radio("統計量", [MEAN_SD, MEDIAN_IQR], key="m_stat")
    show_ctrl = c4.checkbox("対照群も表示", value=False, key="m_ctrl")

    summs, pts, x_order = [], [], []
    for name, (b, conds, control, cpds) in bases.items():
        sub = b[b[GROUP_COL] == group]
        if sub.empty:
            continue
        cpds = detected(sub, cpds)
        s, p = la.summary_values(sub, cpds, kind, control, stat, [COND_COL])
        s.insert(0, DATASET_COL, name)
        p.insert(0, DATASET_COL, name)
        s["対照群"] = control
        if not show_ctrl:
            s, p = s[s[COND_COL] != control], p[p[COND_COL] != control]
        summs.append(s)
        pts.append(p)
        x_order += [c for c in conds if c not in x_order and (show_ctrl or c != control)]
    if not summs:
        st.warning(f"{group} のデータがありません")
        return
    summ = pd.concat(summs, ignore_index=True)
    points = pd.concat(pts, ignore_index=True)
    if summ.empty:
        st.info("対照群以外の condition がありません。condition を追加するか「対照群も表示」をオンにしてください。")
        return
    used = [n for n in ds_names if n in set(summ[DATASET_COL])]
    compounds = sorted(summ.loc[summ["n"] > 0, CPD_COL].unique(), key=la.natural_key)

    with tab_tbl:
        st.caption(f"希釈グループ: {group} ・ 値: {kind} ・ 統計量: {stat}")
        out = summ[[DATASET_COL, "対照群", COND_COL, CPD_COL, "n", "center", "lower", "upper"]].rename(
            columns={"center": stat_label(stat), "lower": "下端", "upper": "上端"})
        st.dataframe(out, hide_index=True, height=400)
        st.download_button("要約表 (long 形式) をダウンロード", to_csv_bytes(out, na_rep), "integrated_summary.csv",
                           "text/csv", key="_dl_m_summary")
        wide = summ.pivot_table(index=CPD_COL, columns=[DATASET_COL, COND_COL], values="center")
        st.download_button(f"{stat_label(stat)} の行列 (化合物 x データセット/condition) をダウンロード",
                           to_csv_bytes(wide, na_rep, index=True), "integrated_matrix.csv", "text/csv", key="_dl_m_matrix")

    with tab_bar:
        c1, c2 = st.columns([4, 1])
        with c1:
            sel = compound_select("m_sel", compounds, default=compounds[:1], max_selections=40)
        show_points = c2.checkbox("各サンプルの点を重ねる", value=True, key="m_pts")
        if sel:
            s = summ[summ[CPD_COL].isin(sel)]
            p = points[points[CPD_COL].isin(sel)]
            st.plotly_chart(bar_summary(s, p, COND_COL, x_order, DATASET_COL, used, pal,
                                        f"{kind} ({stat_label(stat)})", stat_label(stat), show_points, ref_value(kind)),
                            width="stretch")
            st.caption(caption_for(kind, stat) + " 色 = データセット。")

    with tab_hm:
        hm_cpds = compound_picker("m_hm", compounds)
        c1, c2, c3 = st.columns([4, 1, 1])
        with c2:
            cmap = colormap_select("m_hm_cmap")
        limit = c3.number_input("色の上限 |log2FC| (0 = 自動)", 0.0, 20.0, 0.0, 0.5, key="m_hm_lim",
                                disabled=kind != LOG2FC)
        mats, hovers = {}, {}
        for name in used:
            s = summ[(summ[DATASET_COL] == name) & summ[CPD_COL].isin(hm_cpds)]
            conds = [c for c in bases[name][1] if c in set(s[COND_COL])]
            mats[name] = s.pivot(index=COND_COL, columns=CPD_COL, values="center").reindex(conds)
            hovers[name] = summary_hover(s, stat, COND_COL).reindex(conds)
        vals = np.concatenate([m.values.ravel() for m in mats.values()])
        cs, title, mid, zr = colorscale_for(kind, pal, vals, limit, cmap)
        st.plotly_chart(heatmap_grid(mats, cs, f"{title}<br>({stat_label(stat)})", mid, zr, hovers), width="stretch")
        st.caption("パネル = データセット、列 = condition。空白は未検出 (または対照群で未検出)。"
                   "色の上限を超える値は端の色で表示されます (ツールチップには実際の値)。 " + caption_for(kind, stat))


if mode == SINGLE:
    single_mode(datasets[0])
else:
    multi_mode(datasets)

with st.sidebar:
    st.header("解析設定の保存")
    saver(sources)
