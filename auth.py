"""パスワードによるアクセス制限。

パスワードは .streamlit/secrets.toml (ローカル) または Streamlit Community Cloud の Secrets に書く:

    password = "共通のパスワード"

    # 利用者ごとに分ける場合 (どちらか一方、または両方)
    [passwords]
    yamada = "パスワード1"
    suzuki = "パスワード2"

Secrets にパスワードが無い場合、手元の PC では制限なしで起動する。
Streamlit Community Cloud 上で Secrets が未設定のときは、誤って誰でも使える状態にならないよう起動を止める。
"""
import hmac
import os
import time

import streamlit as st


def _secrets():
    try:
        return {"password": st.secrets.get("password"), "passwords": dict(st.secrets.get("passwords", {}))}
    except Exception:  # secrets.toml が無い
        return {"password": None, "passwords": {}}


def _on_community_cloud():
    # Community Cloud はアプリを /mount/src/ 以下に置き、ユーザー appuser で動かす
    return os.getcwd().startswith("/mount/src") or os.environ.get("HOME") == "/home/appuser"


def require_password():
    """認証済みでなければログイン画面を出して処理を止める。"""
    sec = _secrets()
    if not sec["password"] and not sec["passwords"]:
        if _on_community_cloud():
            st.error("パスワードが設定されていません。Streamlit Community Cloud のアプリ設定 → Secrets に "
                     "password = \"...\" を追加してください。")
            st.stop()
        return  # 手元の PC: 制限なし
    if st.session_state.get("_auth_ok"):
        with st.sidebar:
            who = st.session_state.get("_auth_user")
            if st.button("ログアウト" + (f" ({who})" if who else ""), key="_btn_logout"):
                st.session_state.clear()
                st.rerun()
        return

    st.title("LC-MS/MS 定量解析")
    with st.form("_login"):
        user = st.text_input("ユーザー名", key="_auth_name") if sec["passwords"] else ""
        pw = st.text_input("パスワード", type="password", key="_auth_pw")
        submitted = st.form_submit_button("ログイン")
    if submitted:
        ok = False
        if sec["passwords"] and user in sec["passwords"]:
            ok = hmac.compare_digest(pw.encode(), str(sec["passwords"][user]).encode())
        if not ok and sec["password"]:
            ok = hmac.compare_digest(pw.encode(), str(sec["password"]).encode())
        if ok:
            st.session_state["_auth_ok"] = True
            st.session_state["_auth_user"] = user if user in sec["passwords"] else ""
            st.rerun()
        time.sleep(1.5)  # 総当たりを遅らせる
        st.error("ユーザー名またはパスワードが違います")
    st.stop()
