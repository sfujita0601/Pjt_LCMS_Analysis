"""Streamlit 画面で共通に使う部品。"""
import numpy as np
import streamlit as st

from lcms_analysis import FC, GROUP_PRESETS, LOG2FC
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


def available_groups(compounds):
    """選択に使えるグループ: よく使う分類 (データにある化合物が 2 つ以上) + 変数グループ タブで作ったもの。"""
    have = set(compounds)
    groups = {}
    for name, members in GROUP_PRESETS.items():
        m = [x for x in members if x in have]
        if len(m) >= 2:
            groups[f"{name} ({len(m)})"] = m
    for name, members in (st.session_state.get("_var_groups") or {}).items():
        m = [x for x in members if x in have]
        if m:
            groups[f"{name} ({len(m)})"] = m
    return groups


def _group_select(container, key, compounds, disabled=False):
    groups = available_groups(compounds)
    if not groups:
        return []
    gsel = container.multiselect("グループでまとめて選ぶ", list(groups), key=f"{key}_cgrp", disabled=disabled,
                                 help="アミノ酸 (タンパク質構成 20 種) などの分類や、変数グループ タブで作ったグループ")
    return [m for g in gsel for m in groups.get(g, [])]


def compound_select(key, compounds, default=None, label="化合物", max_selections=None):
    """図に描く化合物を選ぶ (個別 + グループ)。選んだ順 (グループは分類の順) のリストを返す。"""
    c1, c2 = st.columns([3, 2])
    sel = c1.multiselect(label, compounds, default=default, key=key, placeholder="化合物を選択 (名前の一部で検索できます)")
    out = list(dict.fromkeys(list(sel) + _group_select(c2, key, compounds)))
    if max_selections and len(out) > max_selections:
        st.warning(f"表示できるのは {max_selections} 個までです。最初の {max_selections} 個を表示します。")
        out = out[:max_selections]
    return out


def compound_picker(key, compounds, label="対象の化合物"):
    """解析の対象にする化合物を選ぶ (すべて / 選んだものだけ / 選んだものを除く)。"""
    c1, c2 = st.columns([1, 3])
    mode = c1.radio(label, [PICK_ALL, PICK_ONLY, PICK_EXCLUDE], key=f"{key}_cmode")
    sel = c2.multiselect("化合物", compounds, key=f"{key}_csel", disabled=mode == PICK_ALL,
                         placeholder="化合物を選択 (名前の一部を入力して検索できます)")
    sel = list(dict.fromkeys(list(sel) + _group_select(c2, key, compounds, disabled=mode == PICK_ALL)))
    if mode == PICK_ONLY:
        if not sel:
            c2.caption("化合物を選ぶまでは、すべての化合物を使います。")
            return list(compounds)
        return [c for c in compounds if c in sel]
    if mode == PICK_EXCLUDE:
        return [c for c in compounds if c not in sel]
    return list(compounds)
