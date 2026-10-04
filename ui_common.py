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


# ---------------------------------------------------------------- 図の表示・保存
FONTS = ["既定", "Arial", "Helvetica", "Times New Roman", "Noto Sans JP", "Hiragino Sans", "Yu Gothic", "Meiryo",
         "その他 (入力)"]
SAVE_FORMATS = {"PNG": "png", "PDF": "pdf", "SVG": "svg", "JPEG": "jpeg"}
_chart_count = [0]


def figure_settings():
    """サイドバーの「図の設定」。値は session_state に入り、show_chart が参照する。"""
    with st.sidebar.expander("図の設定 (フォント・背景・保存)"):
        font = st.selectbox("フォント", FONTS, key="fig_font",
                            help="日本語を含むラベルには Noto Sans JP / Hiragino Sans などの日本語フォントを選んでください")
        if font == "その他 (入力)":
            st.text_input("フォント名", key="fig_font_custom", placeholder="例: Source Han Sans")
        c1, c2 = st.columns(2)
        c1.number_input("軸ラベルの大きさ (0 = 既定)", 0, 48, 0, key="fig_title_size")
        c2.number_input("目盛り数字の大きさ (0 = 既定)", 0, 48, 0, key="fig_tick_size")
        c1.number_input("凡例の大きさ (0 = 既定)", 0, 48, 0, key="fig_legend_size")
        c2.number_input("図タイトルの大きさ (0 = 既定)", 0, 48, 0, key="fig_head_size")
        st.radio("画面の背景", ["テーマに合わせる", "白"], key="fig_bg", horizontal=True)
        st.divider()
        st.selectbox("保存形式", list(SAVE_FORMATS), key="fig_format",
                     help="各図の「図を保存」で使う形式。PDF・SVG は拡大しても劣化しないベクター形式")
        st.radio("保存時の背景", ["白", "透明", "画面と同じ"], key="fig_save_bg", horizontal=True)
        st.number_input("解像度の倍率 (PNG・JPEG)", 1, 6, 3, key="fig_scale")


def _font_family():
    ss = st.session_state
    f = ss.get("fig_font", "既定")
    if f == "その他 (入力)":
        f = ss.get("fig_font_custom") or "既定"
    return None if f == "既定" else f


def style_figure(fig):
    """フォントの種類と大きさ・背景を図に反映する (0 / 既定 の項目は変えない)。"""
    ss = st.session_state
    family = _font_family()
    if family:
        fig.update_layout(font_family=family)
    ts, ks = ss.get("fig_title_size", 0), ss.get("fig_tick_size", 0)
    for upd in (fig.update_xaxes, fig.update_yaxes):
        if ts:
            upd(title_font_size=ts)
        if ks:
            upd(tickfont_size=ks)
    if ks:
        fig.update_coloraxes(colorbar_tickfont_size=ks)
        fig.update_traces(colorbar_tickfont_size=ks, selector=dict(type="heatmap"))
    if ss.get("fig_legend_size", 0):
        fig.update_layout(legend_font_size=ss["fig_legend_size"])
    if ss.get("fig_head_size", 0):
        fig.update_layout(title_font_size=ss["fig_head_size"])
        fig.update_annotations(font_size=ss["fig_head_size"])  # サブプロットのタイトル
    return fig


def _white(fig):
    return fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                             font_color="#0b0b0b")


def export_figure(fig, fmt, background="白", scale=3):
    """図を画像・PDF のバイト列にする (kaleido を使う)。"""
    import plotly.graph_objects as go

    out = go.Figure(fig)
    if background == "白":
        _white(out)
    elif background == "透明":
        out.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    width = out.layout.width or 1100
    height = out.layout.height or 500
    return out.to_image(format=fmt, width=width, height=height, scale=scale if fmt in ("png", "jpeg") else 1)


def show_chart(container, fig, key=None, width="stretch", **kwargs):
    """st.plotly_chart の代わり: 図の設定を反映して表示し、「図を保存」を付ける。"""
    ss = st.session_state
    _chart_count[0] += 1
    key = key or f"chart{_chart_count[0]}"
    style_figure(fig)
    white = ss.get("fig_bg") == "白"
    if white:
        _white(fig)
    fmt_name = ss.get("fig_format", "PNG")
    fmt = SAVE_FORMATS[fmt_name]
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else key
    fname = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(title))[:60] or "figure"
    config = {"displaylogo": False,
              "toImageButtonOptions": {"format": fmt if fmt in ("png", "svg", "jpeg") else "png",
                                       "filename": fname, "scale": ss.get("fig_scale", 3)}}
    container.plotly_chart(fig, width=width, key=key, config=config, theme=None if white else "streamlit", **kwargs)
    with container.popover(f"図を保存 ({fmt_name})", icon=":material/download:"):
        st.caption("形式・背景・解像度はサイドバーの「図の設定」で変えられます。")
        if st.button("作成", key=f"_btn_save_{key}"):
            try:
                ss[f"_img_{key}"] = export_figure(fig, fmt, ss.get("fig_save_bg", "白"), ss.get("fig_scale", 3))
            except Exception as e:  # Chrome が無い環境など
                ss.pop(f"_img_{key}", None)
                st.error(f"画像を作成できません: {e} (図の右上のカメラのボタンでも PNG / SVG で保存できます)")
        data = ss.get(f"_img_{key}")
        if data:
            st.download_button(f"{fname}.{fmt} をダウンロード", data, f"{fname}.{fmt}", key=f"_dl_save_{key}")


def reset_chart_counter():
    _chart_count[0] = 0


# ---------------------------------------------------------------- 出力の登録 (まとめて出力用)
def register_output(name, df):
    """各タブで作った表を登録する。サイドバーの「解析結果をまとめて出力」で zip にまとめる。"""
    st.session_state.setdefault("_outputs", {})[name] = df


def reset_outputs():
    st.session_state["_outputs"] = {}


def app_version():
    """アプリの版: git のコミット (取得できれば) と pyproject の version。"""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent
    ver = "unknown"
    try:
        import tomllib

        ver = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    except Exception:
        pass
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
                               timeout=5).stdout.strip()
        if commit:
            ver += f" (git {commit}{' + 未コミットの変更' if dirty else ''})"
    except Exception:
        pass
    return ver
