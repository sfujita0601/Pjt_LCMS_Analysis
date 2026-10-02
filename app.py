"""LC-MS/MS 定量結果の整形・QC・可視化・統計解析 GUI (Streamlit)

起動:
    uv run streamlit run app.py
"""
import streamlit as st

from auth import require_password

st.set_page_config(page_title="LC-MS/MS 定量解析", layout="wide")
require_password()  # パスワードが設定されていれば、ログインするまでここで止まる
page = st.navigation([
    st.Page("views/analysis.py", title="解析", icon=":material/analytics:", default=True),
    st.Page("views/docs.py", title="解析手法の解説", icon=":material/menu_book:"),
])
page.run()
