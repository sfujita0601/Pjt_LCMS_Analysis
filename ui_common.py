"""Streamlit 画面で共通に使う部品。"""
import numpy as np
import streamlit as st

from lcms_analysis import FC, LOG2FC
from lcms_plots import PALETTE, colormap_options, resolve_colormap, scale


def theme():
    try:
        return PALETTE["dark" if st.context.theme.type == "dark" else "light"]
    except Exception:
        return PALETTE["light"]


def to_csv_bytes(df, na="", index=False):
    return df.to_csv(index=index, na_rep=na).encode("utf-8-sig")


def stable_editor(key, init_df, **kwargs):
    """編集内容が再描画で消えない data_editor。

    初期表は session_state に固定して渡す (毎回作り直すと編集がリセットされるため)。
    reset_editor(key, df) で中身を差し替えられる。
    """
    ss = st.session_state
    if f"{key}::init" not in ss:
        ss[f"{key}::init"], ss[f"{key}::ver"] = init_df, 0
    edited = st.data_editor(ss[f"{key}::init"], key=f"{key}::editor::{ss[f'{key}::ver']}", **kwargs)
    ss[f"{key}::cur"] = edited
    return edited


def reset_editor(key, df):
    ss = st.session_state
    ss[f"{key}::init"] = df
    ss[f"{key}::ver"] = ss.get(f"{key}::ver", 0) + 1


def current_editor(key):
    ss = st.session_state
    return ss.get(f"{key}::cur", ss.get(f"{key}::init"))


def colormap_select(key, label="カラーマップ"):
    return st.selectbox(label, colormap_options(), key=key,
                        help="「既定」は値に応じて 青 (連続) / 青-灰-赤 (発散)。末尾 _r は反転")


def colorscale_for(kind, pal, values, limit=0.0, cmap="既定"):
    """(colorscale, colorbar タイトル, 中心, 範囲) を返す。
    log2FC の色範囲は ±limit (0 なら |値| の 98 パーセンタイルで、外れ値に引っ張られないようにする)。
    cmap が "既定" なら、値の種類に応じて 連続 (青) / 発散 (青-灰-赤) を使う。"""
    div, seq = resolve_colormap(cmap, scale(pal["div"])), resolve_colormap(cmap, scale(pal["seq"]))
    if kind == LOG2FC:
        m = symmetric_limit(values, limit)
        return div, "log2 FC", 0, (-m, m)
    if kind == FC:
        return div, "FC", 1, None
    if kind == "Z スコア (化合物ごと)":
        return div, "Z", 0, None
    return seq, "log10 濃度" if kind == "log10" else "濃度", None, None


def symmetric_limit(values, limit=0.0):
    values = np.asarray(values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    return limit or (float(np.percentile(finite, 98)) if finite.size else 1.0) or 1.0


PICK_ALL, PICK_ONLY, PICK_EXCLUDE = "すべて", "選んだ化合物のみ", "選んだ化合物を除外"


def compound_picker(key, compounds, label="対象の化合物"):
    """解析の対象にする化合物を選ぶ (すべて / 選んだものだけ / 選んだものを除く)。"""
    c1, c2 = st.columns([1, 3])
    mode = c1.radio(label, [PICK_ALL, PICK_ONLY, PICK_EXCLUDE], key=f"{key}_cmode")
    sel = c2.multiselect("化合物", compounds, key=f"{key}_csel", disabled=mode == PICK_ALL,
                         placeholder="化合物を選択 (名前の一部を入力して検索できます)")
    groups = st.session_state.get("_var_groups") or {}
    if groups:
        gsel = c2.multiselect("変数グループでまとめて選ぶ", list(groups), key=f"{key}_cgrp", disabled=mode == PICK_ALL)
        sel = list(dict.fromkeys(list(sel) + [m for g in gsel for m in groups.get(g, [])]))
    if mode == PICK_ONLY:
        if not sel:
            c2.caption("化合物を選ぶまでは、すべての化合物を使います。")
            return list(compounds)
        return [c for c in compounds if c in sel]
    if mode == PICK_EXCLUDE:
        return [c for c in compounds if c not in sel]
    return list(compounds)
