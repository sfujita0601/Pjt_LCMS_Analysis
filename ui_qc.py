"""QC タブ: 検量線範囲による希釈の選択 / STD の正確さ / 測定順ドリフト。"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import lcms_qc as qc
import lcms_stats as ls
from lcms_analysis import COND_COL, natural_key
from lcms_plots import color_map, scale
from make_conc_table import DILUTION_COL, LABEL_COL, is_std
from ui_common import current_editor, reset_editor, show_chart, stable_editor, to_csv_bytes

CHOICES = [qc.CHOICE_AUTO, qc.CHOICE_AUTO_SAMPLE, qc.SRC_NONE, qc.SRC_DIL]


def dilution_factor(values):
    m = [re.match(r"x(\d+(?:\.\d+)?)", d) for d in values[DILUTION_COL].unique() if d]
    return float(m[0].group(1)) if m and m[0] else 10.0


def calibration_choices(ds, full, raw, compounds, editable=True):
    """化合物ごとの検量線範囲と希釈の選択。(範囲の表, {化合物: 選択}) を返す。"""
    rng = qc.calibration_ranges(full)
    summ = qc.dilution_summary(raw[~raw[LABEL_COL].map(is_std)], compounds, rng).set_index("化合物")
    init = pd.DataFrame({
        "化合物": compounds,
        "濃度レベル": [rng["濃度レベル"].get(c, "") for c in compounds],
        "1点検量": [bool(rng["一点検量"].get(c, False)) for c in compounds],
        "下限": summ["下限"].values, "上限": summ["上限"].values,
        "検出数": summ["検出数 (希釈なし)"].values, "上限超過": summ["上限超過"].values,
        "下限未満": summ["下限未満"].values, "最大値 (希釈なし)": summ["最大値 (希釈なし)"].round(3).values,
        "選択": qc.CHOICE_AUTO,
    })
    if not editable:
        return init.set_index("化合物")[["下限", "上限"]], {}
    key = f"calib::{ds}"
    c1, c2, c3 = st.columns([2, 1, 3])
    bulk = c1.selectbox("全化合物の選択を一括変更", CHOICES, key=f"{key}::bulk")
    if c2.button("一括変更", key=f"_btn_{key}::bulk"):
        cur = current_editor(key)
        if cur is not None:
            cur = cur.copy()
            cur["選択"] = bulk
            reset_editor(key, cur)
            st.rerun()
    c3.caption("「自動 (化合物単位)」: 希釈なしの濃度が 1 サンプルでも上限を超えた化合物は、全サンプルで希釈測定の値 × 希釈倍率を採用 (推奨)。"
               "「自動 (サンプル単位)」: 上限を超えたサンプルだけ希釈測定を採用。希釈なしと希釈測定に系統差がある場合 (下の「希釈の直線性」) は"
               "見かけの差が生じるので注意してください。下限・上限は編集できます。")
    edited = stable_editor(
        key, init, hide_index=True, num_rows="fixed", height=420,
        disabled=["化合物", "濃度レベル", "1点検量", "検出数", "上限超過", "下限未満", "最大値 (希釈なし)"],
        column_config={
            "選択": st.column_config.SelectboxColumn("選択", options=CHOICES, required=True),
            "下限": st.column_config.NumberColumn("下限", format="%.3g"),
            "上限": st.column_config.NumberColumn("上限", format="%.3g"),
        },
    )
    st.download_button("設定を CSV でダウンロード", to_csv_bytes(edited), f"{ds}_calibration_choice.csv", "text/csv",
                       key=f"_dl_{key}")
    return edited.set_index("化合物")[["下限", "上限"]], dict(zip(edited["化合物"], edited["選択"]))


def merged_table(values, raw, compounds, ranges, choices):
    no_std = ~values[LABEL_COL].map(is_std)
    return qc.merge_dilutions(values[no_std], raw[~raw[LABEL_COL].map(is_std)], compounds, ranges, choices,
                              dilution_factor(values))


def calibration_tab(ds, full, values, raw, compounds, na_rep):
    st.caption("検量線の範囲 (検量点に使われた STD の設定濃度の最小〜最大) と、希釈なし / 希釈測定の採用を化合物ごとに決めます。"
               "結果はサイドバーの「希釈の扱い」で「検量線範囲で統合」を選ぶと以降の解析に使われます。")
    ranges, choices = calibration_choices(ds, full, raw, compounds)
    merged, src, flag = merged_table(values, raw, compounds, ranges, choices)
    st.subheader("統合結果")
    used = (src == qc.SRC_DIL).sum()
    st.markdown(f"希釈測定を採用したセル: **{int(used.sum())}** / {src.size} ・ "
                f"希釈測定を 1 サンプル以上で採用した化合物: **{int((used > 0).sum())}**")
    counts = flag.apply(pd.Series.value_counts).T.fillna(0).astype(int)
    st.dataframe(counts.loc[counts.drop(columns=["範囲内"], errors="ignore").sum(axis=1) > 0], height=250)
    t1, t2, t3 = st.tabs(["統合後の値", "採用元", "範囲判定"])
    with t1:
        st.dataframe(merged, hide_index=True, height=360)
        st.download_button("統合後の表 (CSV)", to_csv_bytes(merged, na_rep), f"{ds}_merged.csv", "text/csv",
                           key=f"_dl_merged::{ds}")
    colors = {qc.SRC_DIL: "background-color: rgba(235,104,52,0.25)", "上限超過": "background-color: rgba(227,73,72,0.30)",
              "下限未満": "background-color: rgba(42,120,214,0.20)", "未検出": "color: #898781"}
    with t2:
        st.dataframe(src.style.map(lambda v: colors.get(v, "")), height=360)
    with t3:
        st.dataframe(flag.style.map(lambda v: colors.get(v, "")), height=360)
    dilution_linearity_section(raw, compounds, dilution_factor(values))
    return merged


def dilution_linearity_section(raw, compounds, factor):
    st.subheader("希釈の直線性")
    tbl, pairs = qc.dilution_linearity(raw[~raw[LABEL_COL].map(is_std)], compounds, factor)
    if tbl.empty:
        st.info("希釈なし・希釈測定の両方で検出されたサンプルが 3 つ以上ある化合物がありません")
        return
    st.caption(f"両方で検出されたサンプルについて (希釈測定 × {factor:g}) / 希釈なし を比べます (IS 補正前の算出濃度)。"
               "比が 1 なら希釈しても同じ濃度が得られています。±15% は ICH M10 の dilution integrity の目安です。"
               "外れる場合は、応答の非直線性 (検量線範囲外での外挿)、マトリックス効果、測定までの時間による変化を疑います。")
    tbl = tbl.sort_values("比の中央値", key=lambda v: (v - 1).abs(), ascending=False)
    c1, c2 = st.columns([3, 2])
    c1.dataframe(tbl.round(3), hide_index=True, height=360)
    c1.download_button("希釈の直線性 (CSV)", to_csv_bytes(tbl), "dilution_linearity.csv", "text/csv", key="_dl_lin")
    with c2:
        cpd = st.selectbox("化合物", tbl["化合物"], key="lin_cpd")
        p = pairs[pairs["化合物"] == cpd]
        mx = float(max(p["希釈なし"].max(), p["希釈測定 x 倍率"].max())) * 1.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="y = x",
                                 line=dict(color="#898781", dash="dot", width=1)))
        fig.add_trace(go.Scatter(x=p["希釈なし"], y=p["希釈測定 x 倍率"], mode="markers+text", text=p[LABEL_COL],
                                 textposition="top center", textfont=dict(size=9), name="サンプル",
                                 marker=dict(size=9, color="#2a78d6"),
                                 hovertemplate="%{text}<br>希釈なし %{x:.4g}<br>希釈測定 x 倍率 %{y:.4g}<extra></extra>"))
        fig.update_layout(height=360, xaxis_title="希釈なし", yaxis_title=f"希釈測定 x {factor:g}", title=cpd,
                          showlegend=False, margin=dict(t=40))
        show_chart(st, fig, width="stretch")


# ---------------------------------------------------------------- STD の正確さ
def accuracy_tab(full, pal):
    acc = qc.std_accuracy(full)
    summ = qc.accuracy_summary(acc)
    st.caption("判定基準: 設定濃度に対する正確さが 100 ± 15% 以内 (最低濃度の STD は ± 20%)、"
               "かつ検量線の標準の 75% 以上が基準内 (US FDA Bioanalytical Method Validation, 2018 / ICH M10, 2022)。"
               "1 点検量の化合物は正確さが定義上 100% になるため評価できません。")
    c1, c2 = st.columns([1, 3])
    show_one = c1.checkbox("1 点検量の化合物も表示", value=False, key="acc_one")
    only_fail = c1.checkbox("基準を満たさない化合物のみ", value=False, key="acc_fail")
    view = summ if show_one else summ[summ["検量線基準 (≥75%)"] != "評価不可 (1点検量)"]
    if only_fail:
        view = view[view["検量線基準 (≥75%)"] == "満たさない"]
    c2.dataframe(view, hide_index=True, height=300)
    c2.download_button("STD の正確さ (long 形式, CSV)",
                       to_csv_bytes(acc[["compound", "file", "set_conc", "conc", "area", "accuracy", "error", "許容幅%", "判定"]]),
                       "std_accuracy.csv", "text/csv", key="_dl_acc")

    cpds = sorted(view["化合物"], key=natural_key)
    if not cpds:
        return
    files = acc.drop_duplicates("file").sort_values("datetime")["file"].tolist()
    M = acc.pivot_table(index="compound", columns="file", values="accuracy").reindex(index=cpds, columns=files)
    J = acc.pivot_table(index="compound", columns="file", values="判定", aggfunc="first").reindex(index=cpds, columns=files)
    S = acc.pivot_table(index="compound", columns="file", values="set_conc").reindex(index=cpds, columns=files)
    short = [re.sub(r"^\d{8}_", "", f).replace(".lcd", "") for f in files]
    fig = go.Figure(go.Heatmap(
        z=M.values, x=short, y=cpds, colorscale=scale(pal["div"]), zmid=100, zmin=50, zmax=150, xgap=1, ygap=1,
        customdata=np.dstack([J.fillna("").values, S.values]),
        colorbar=dict(title="正確さ %"),
        hovertemplate="%{y}<br>%{x}<br>設定濃度 %{customdata[1]}<br>正確さ %{z:.1f}%<br>%{customdata[0]}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickfont_size=10)
    fig.update_layout(height=max(400, 16 * len(cpds) + 160), margin=dict(l=220, t=30),
                      title="STD ごとの正確さ (%) ・ 色の範囲 50–150% (100% が中央)")
    show_chart(st, fig, width="stretch")

    st.subheader("検量線")
    cpd = st.selectbox("化合物", cpds, key="acc_cpd")
    a = acc[(acc["compound"] == cpd) & acc["set_conc"].notna()]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["設定濃度 vs 算出濃度", "設定濃度 vs 面積"])
    mx = float(np.nanmax([a["set_conc"].max(), a["conc"].max()])) * 1.1
    xs = np.array([0, mx])
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color=pal["line"], dash="dot", width=1), name="y = x"))
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[xs * 1.15, xs[::-1] * 0.85], fill="toself", mode="none",
                             fillcolor="rgba(137,135,129,0.15)", name="± 15%"))
    cmap = {"合格": pal["series"][0], "不合格": pal["series"][7], "評価不可 (1点検量)": "#898781", "未算出": "#898781"}
    for j, g in a.groupby("判定"):
        fig.add_trace(go.Scatter(x=g["set_conc"], y=g["conc"], mode="markers", name=j,
                                 marker=dict(size=10, color=cmap.get(j, "#898781")), text=g["file"],
                                 hovertemplate="%{text}<br>設定 %{x}<br>算出 %{y:.3g}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["set_conc"], y=g["area"], mode="markers", showlegend=False,
                                 marker=dict(size=10, color=cmap.get(j, "#898781")), text=g["file"],
                                 hovertemplate="%{text}<br>設定 %{x}<br>面積 %{y:.4g}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title="設定濃度")
    fig.update_yaxes(title="算出濃度", row=1, col=1)
    fig.update_yaxes(title="面積", row=1, col=2)
    fig.update_layout(height=420)
    show_chart(st, fig, width="stretch")


# ---------------------------------------------------------------- ドリフト
def drift_tab(full, is_name, pal, cond_map):
    inj = qc.injection_order(full)
    st.caption("分析日時から測定順を求め、値の経時変化を確認します。本来はプール QC 試料を一定間隔で測定して評価します"
               " (Dunn et al., 2011; Broadhurst et al., 2018)。QC が無い場合は内部標準や STD の繰り返し測定で代用します。")
    c1, c2, c3 = st.columns([3, 1, 1])
    cpds = sorted(full["compound"].unique(), key=natural_key)
    sel = c1.multiselect("化合物", cpds, default=[is_name] if is_name in cpds else cpds[:1], max_selections=8,
                         key="drift_cpds")
    value = c2.radio("値", ["面積", "濃度 (補正前)"], key="drift_val")
    color_by = c3.radio("色分け", ["種別", "condition"], key="drift_color")
    col = "area" if value == "面積" else "conc"

    df = full.merge(inj[["file", "order"]], on="file")
    df["種別"] = np.where(df["is_std"], "STD", np.where(df[DILUTION_COL] == "", qc.SRC_NONE, df[DILUTION_COL]))
    df[COND_COL] = df[LABEL_COL].map(cond_map).fillna("未設定")
    key = "種別" if color_by == "種別" else COND_COL
    cats = list(dict.fromkeys(["STD", qc.SRC_NONE] + sorted(df["種別"].unique()))) if key == "種別" \
        else list(dict.fromkeys(list(dict.fromkeys(cond_map.values())) + ["未設定"]))
    cmap = color_map(cats, pal)

    if sel:
        fig = make_subplots(rows=len(sel), cols=1, subplot_titles=sel, shared_xaxes=True,
                            vertical_spacing=0.25 / max(1, len(sel)))
        for r, c in enumerate(sel, start=1):
            d = df[df["compound"] == c]
            for k in cats:
                g = d[d[key] == k]
                if g.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=g["order"], y=g[col], mode="markers", name=k, legendgroup=k, showlegend=r == 1,
                    marker=dict(size=9, color=cmap[k], line=dict(width=1, color="rgba(255,255,255,0.8)")),
                    text=g["file"], customdata=g["datetime"].dt.strftime("%m/%d %H:%M"),
                    hovertemplate="%{text}<br>%{customdata}<br>測定順 %{x}<br>%{y:.4g}<extra></extra>",
                ), row=r, col=1)
            # サンプルのみで希釈グループごとの回帰直線
            for dil, g in d[~d["is_std"]].groupby(DILUTION_COL):
                g = g[np.isfinite(g[col])]
                if len(g) >= 4:
                    k_, b_ = np.polyfit(g["order"], g[col], 1)
                    xs = np.array([g["order"].min(), g["order"].max()])
                    fig.add_trace(go.Scatter(x=xs, y=k_ * xs + b_, mode="lines", showlegend=False, hoverinfo="skip",
                                             line=dict(color=pal["line"], width=1, dash="dash")), row=r, col=1)
            fig.update_yaxes(title=value, row=r, col=1)
        fig.update_xaxes(title="測定順", row=len(sel), col=1)
        fig.update_layout(height=max(380, 280 * len(sel)), legend_title_text=key)
        show_chart(st, fig, width="stretch")
        st.caption("破線 = サンプルの値に当てはめた回帰直線 (希釈グループ別)。")

    st.subheader("全化合物のドリフト指標")
    tbl = qc.drift_table(full, inj, col)
    if tbl.empty:
        return
    tbl["q (BH)"] = ls.adjust_p(tbl["p"], "fdr_bh").values
    thr = st.slider("|ρ| の目安", 0.0, 1.0, 0.5, 0.05, key="drift_thr")
    tbl["判定"] = np.where((tbl["Spearman ρ"].abs() >= thr) & (tbl["q (BH)"] <= 0.05), "ドリフトの疑い", "")
    st.dataframe(tbl.sort_values("Spearman ρ", key=np.abs, ascending=False).round(4), hide_index=True, height=360)
    st.download_button("ドリフト指標 (CSV)", to_csv_bytes(tbl), "drift.csv", "text/csv", key="_dl_drift")
    if cond_map:
        samp = inj[~inj["is_std"]].assign(c=lambda x: x[LABEL_COL].map(cond_map))
        st.caption("注意: condition と測定順が偏っていると、ドリフトと群間差を区別できません。測定順はランダム化が推奨されます。"
                   " 測定順の範囲: " + " ・ ".join(f"{c}: {g['order'].min()}–{g['order'].max()} (平均 {g['order'].mean():.1f})"
                                           for c, g in samp.dropna(subset=["c"]).groupby("c")))
