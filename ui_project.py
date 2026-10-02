"""解析設定の保存と読み込み (再解析の再現用)。

保存するもの: 画面のすべての設定 (キー付きのウィジェットの値)、条件設定・検量線の選択・比の定義・KEGG 対応表の表、
データファイル名と SHA-256、主要ライブラリのバージョン。データそのものは保存しない (同じファイルを読み込んで使う)。
ボタン・アップローダー・ダウンロードのキーは "_" で始め、保存の対象から外している。
"""
import hashlib
import json
from datetime import datetime
from importlib import metadata
from pathlib import Path

import pandas as pd
import streamlit as st

FORMAT = "lcms-app-project"
VERSION = 1
PACKAGES = ["streamlit", "pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "umap-learn", "networkx", "plotly"]
# 表 (data_editor) の内部状態。"<名前>::cur" に現在の表が入っている
EDITOR_SUFFIX = "::cur"
SKIP_PARTS = ("::editor::", "::init", "::ver", "cond_init::", "cond_ver::", "map_done::", "FormSubmitter")


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def _encode(v):
    if isinstance(v, Path):
        return {"__path__": v.name}
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    if hasattr(v, "item"):  # numpy の数値
        return v.item()
    raise TypeError


def _decode(v, data_dir):
    if isinstance(v, dict) and "__path__" in v:
        return data_dir / v["__path__"]
    if isinstance(v, list):
        return [_decode(x, data_dir) for x in v]
    return v


def snapshot(sources, external=()):
    """現在の設定を dict にする。sources: [(データセット名, bytes)]、external: [(外部データのファイル名, bytes)]"""
    ss = st.session_state
    widgets, editors = {}, {}
    for k in list(ss.keys()):
        if not isinstance(k, str) or k.startswith(("_", "$$")):
            continue
        if k.startswith("cond_cur::") or k.endswith(EDITOR_SUFFIX):
            df = ss[k]
            if isinstance(df, pd.DataFrame):
                editors[k] = json.loads(df.to_json(orient="split", force_ascii=False))
            continue
        if any(p in k for p in SKIP_PARTS):
            continue
        try:
            widgets[k] = _encode(ss[k])
        except TypeError:
            pass
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return {
        "format": FORMAT, "version": VERSION, "saved_at": datetime.now().isoformat(timespec="seconds"),
        "data": [{"name": n, "sha256": sha256(b), "bytes": len(b)} for n, b in sources],
        "external": [{"name": n, "sha256": sha256(b), "bytes": len(b)} for n, b in external],
        "packages": versions, "widgets": widgets, "editors": editors,
    }


def apply(project, data_dir):
    """読み込んだ設定を session_state に入れる (次の再実行で各ウィジェットに反映される)。"""
    ss = st.session_state
    for k, v in project.get("widgets", {}).items():
        ss[k] = _decode(v, data_dir)
    for k, payload in project.get("editors", {}).items():
        df = pd.DataFrame(payload["data"], columns=payload["columns"])
        if k.startswith("cond_cur::"):
            ds = k.split("::", 1)[1]
            init_key, ver_key = f"cond_init::{ds}", f"cond_ver::{ds}"
        else:
            base = k[: -len(EDITOR_SUFFIX)]
            init_key, ver_key = f"{base}::init", f"{base}::ver"
        ss[init_key] = df
        ss[ver_key] = ss.get(ver_key, 0) + 1  # data_editor のキーを変えて、編集状態を読み込んだ表で作り直す
        ss.pop(k, None)


def loader(data_dir):
    """サイドバーの先頭に置く: 設定ファイルを読み込んだら反映して再実行する。"""
    up = st.file_uploader("解析設定を読み込む (.json)", type=["json"], key="_up_project",
                          help="以前に保存した設定を復元します。データファイルは同じものを選んでください")
    if up is not None and st.session_state.get("_project_done") != up.file_id:
        try:
            project = json.loads(up.getvalue().decode("utf-8"))
            if project.get("format") != FORMAT:
                raise ValueError("このアプリの設定ファイルではありません")
            apply(project, data_dir)
        except Exception as e:
            st.error(f"設定を読み込めません: {e}")
        else:
            st.session_state["_project_done"] = up.file_id
            st.session_state["_project_loaded"] = project
            st.rerun()


def check_data(sources):
    """読み込んだ設定のデータファイルと、今選ばれているデータが一致するか確認する。"""
    project = st.session_state.get("_project_loaded")
    if not project:
        return
    saved = {d["name"]: d["sha256"] for d in project.get("data", [])}
    now = {n: sha256(b) for n, b in sources}
    if saved == now:
        st.success(f"設定を復元しました ({project.get('saved_at', '')} 保存)。データファイルも一致しています。")
    else:
        diff = [n for n in set(saved) | set(now) if saved.get(n) != now.get(n)]
        st.warning("設定を復元しましたが、データファイルが保存時と異なります: " + ", ".join(sorted(diff))
                   + "。同じ結果を再現するには、保存時と同じファイルを選んでください。")


def saver(sources, external=()):
    """サイドバーの末尾に置く: 現在の設定を JSON でダウンロードする。"""
    project = snapshot(sources, external)
    st.download_button("解析設定を保存 (.json)", json.dumps(project, ensure_ascii=False, indent=1).encode("utf-8"),
                       f"lcms_project_{datetime.now():%Y%m%d_%H%M}.json", "application/json", key="_dl_project",
                       help="画面の設定・条件設定・検量線の選択・比の定義などを保存します (データ本体は含みません)")
    st.caption(f"保存される設定: {len(project['widgets'])} 項目 + 表 {len(project['editors'])} 個")
