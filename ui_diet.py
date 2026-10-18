"""タブ: 飼料のアミノ酸組成と血中アミノ酸の変化の比較。"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lcms_diet as ld
import lcms_stats as ls
from lcms_analysis import COND_COL, GROUP_COL
from lcms_plots import color_map
from ui_common import alpha, register_output, show_chart, stable_editor, to_csv_bytes
from ui_external import _LocalFile


def diet_tab(base, groups, conds, control, compounds, pal, data_dir):
    st.caption("飼料ごとのアミノ酸含量の表を読み込み、各群の飼料と対照群の飼料の含量の比 (log2) と、"
               "血中濃度の変化 (log2 幾何平均比と 95% 信頼区間) をアミノ酸ごとに並べます。"
               "表は 1 行 = 1 飼料、1 列目 = 飼料名、残りの列 = アミノ酸の含量 (単位は飼料間でそろえてください)。")
    if control is None:
        st.info("条件設定タブで condition と対照群を設定してください。")
        return
    up = st.file_uploader("飼料組成の表 (CSV / TSV / Excel)", type=["csv", "tsv", "txt", "xlsx", "xlsm"], key="_up_diet")
    local = sorted(p for p in data_dir.glob("*") if p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xlsm")
                   and ("diet" in p.stem.lower() or "飼料" in p.stem))
    pick = st.selectbox("または data/ から選択 (名前に diet / 飼料 を含むファイル)", [None] + local,
                        format_func=lambda p: "選択しない" if p is None else p.name, key="diet_local")
    f = up or (_LocalFile(pick) if pick else None)
    if f is None:
        return
    try:
        diet = ld.read_diet(f.getvalue(), f.name)
    except Exception as e:
        st.error(f"飼料組成の表を読めません: {e}")
        return
    unit = st.text_input("含量の単位 (表示用)", value="g/kg 飼料", key="diet_unit")
    st.dataframe(diet, height=200)
    if "模擬" in f.name or "simulated" in f.name.lower() or "demo" in f.name.lower():
        st.error("模擬データ (架空の飼料組成) です。")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**condition と飼料の対応**")
        names = list(diet.index)
        init = pd.DataFrame({COND_COL: conds, "飼料": [c if c in names else "" for c in conds]})
        cmap_tbl = stable_editor("diet_cond", init, hide_index=True, num_rows="fixed", disabled=[COND_COL],
                                 column_config={"飼料": st.column_config.SelectboxColumn("飼料", options=[""] + names)})
        cond_to_diet = {r[COND_COL]: r["飼料"] for _, r in cmap_tbl.iterrows() if r["飼料"]}
    with c2:
        st.markdown("**飼料の列と化合物の対応** (自動で照合。違う場合は修正)")
        auto = ld.match_compounds(diet.columns, compounds)
        init = pd.DataFrame({"飼料の列": list(diet.columns), "化合物": [auto[c] for c in diet.columns]})
        map_tbl = stable_editor("diet_map", init, hide_index=True, num_rows="fixed", disabled=["飼料の列"],
                                column_config={"化合物": st.column_config.SelectboxColumn("化合物", options=[""] + list(compounds))})
        mapping = {r["飼料の列"]: r["化合物"] for _, r in map_tbl.iterrows() if r["化合物"]}
        unmatched = [c for c in diet.columns if c not in mapping]
        if unmatched:
            st.caption("対応する化合物が無い列 (比較から除外): " + ", ".join(map(str, unmatched)))
    if control not in cond_to_diet:
        st.warning(f"対照群 {control} の飼料を指定してください。")
        return
    others = [c for c in conds if c != control and c in cond_to_diet]
    if not others:
        st.warning("対照群以外の condition に飼料を指定してください。")
        return

    c1, c2, c3, c4 = st.columns(4)
    group = c1.selectbox("希釈グループ", groups, key="diet_group")
    test = c2.selectbox("血中の変化の検定", ls.TESTS, key="diet_test")
    corr = c3.selectbox("多重性補正", list(ls.CORRECTIONS), key="diet_corr")
    q_thr = c4.number_input("補正後 p の閾値", 0.0, 1.0, alpha(), 0.005, format="%.4f", key="diet_q")
    diet_thr = c1.number_input("飼料が「変化した」とみなす |log2 比|", 0.0, 10.0, 0.25, 0.05, key="diet_thr",
                               help="0.25 は約 1.19 倍。分類の目安にだけ使います")
    sub = base[base[GROUP_COL] == group]
    targets = sorted(set(mapping.values()))
    plasma = pd.concat([ls.compare_groups(sub, targets, t, control, test, corr).assign(比較群=t) for t in others],
                       ignore_index=True)
    ratio = ld.diet_log2_ratio(diet, cond_to_diet, control)
    table = ld.combine(ratio, plasma, mapping, q_thr, diet_thr)
    if table.empty:
        st.warning("比較できるアミノ酸がありません (対応づけを確認してください)")
        return

    cmap = color_map(others, pal)
    fig = go.Figure()
    lim = np.nanmax(np.abs(np.r_[table["飼料 log2 比"].values, table["血中 log2 比"].values])) * 1.15 or 1
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="y = x (飼料と同じ比で変化)",
                             line=dict(color="#c3c2b7", dash="dot", width=1), hoverinfo="skip"))
    for cond in others:
        g = table[table["condition"] == cond]
        sig = g["q"] <= q_thr
        fig.add_trace(go.Scatter(
            x=g["飼料 log2 比"], y=g["血中 log2 比"], mode="markers+text", name=f"{cond} vs {control}",
            text=g["化合物"], textposition="top center", textfont=dict(size=9),
            error_y=dict(type="data", symmetric=False, array=g["95% CI 上限"] - g["血中 log2 比"],
                         arrayminus=g["血中 log2 比"] - g["95% CI 下限"], color=cmap[cond], thickness=1, width=0),
            marker=dict(size=10, color=[cmap[cond] if s_ else "white" for s_ in sig], line=dict(width=1.8, color=cmap[cond])),
            customdata=np.stack([g["q"], g["分類"]], axis=-1),
            hovertemplate="%{text}<br>飼料 log2 比 %{x:.3f}<br>血中 log2 比 %{y:.3f}<br>補正後 p %{customdata[0]:.3g}"
                          "<br>%{customdata[1]}<extra>" + cond + "</extra>"))
    fig.add_hline(y=0, line=dict(color="#898781", width=1))
    fig.add_vline(x=0, line=dict(color="#898781", width=1))
    fig.update_xaxes(title=f"飼料中の含量の log2 比 (各群 / 対照群, {unit})", range=[-lim, lim], zeroline=False)
    fig.update_yaxes(title="血中濃度の log2 幾何平均比 (各群 / 対照群)", range=[-lim, lim], zeroline=False,
                     scaleanchor="x")
    fig.update_layout(height=640, title=f"飼料組成 vs 血中アミノ酸 ({group}, {test})", legend_title_text="比較")
    show_chart(st, fig)
    st.caption(f"縦線 = 血中の 95% 信頼区間、塗りつぶし = 補正後 p ≤ {q_thr:g}。点線 (y = x) は「飼料中と同じ比で血中も変化した」場合。"
               "飼料が変わっても血中が動かないアミノ酸は、吸収・代謝・恒常性維持で調節されている可能性があります。")

    st.markdown("**アミノ酸をまたいだ関連** (探索用)")
    assoc = ld.association(table)
    st.dataframe(assoc.round(4), hide_index=True)
    st.caption("各点 (アミノ酸) は独立な観測ではなく、代謝経路でつながっています。p 値は探索の目安として扱ってください。")
    st.markdown("**アミノ酸ごとの比較**")
    summary = table["分類"].value_counts().rename("アミノ酸 x 群の数")
    st.dataframe(summary)
    st.dataframe(table.round(4), hide_index=True, height=320)
    st.download_button("飼料組成と血中の比較 (CSV)", to_csv_bytes(table), "diet_vs_plasma.csv", "text/csv", key="_dl_diet")
    register_output(f"統計/飼料組成と血中の比較 ({group})", table)
    register_output(f"統計/飼料組成と血中の関連 ({group})", assoc)
