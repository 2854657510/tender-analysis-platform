"""RAG 问答 —— D12 任务。"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import api_client  # noqa: E402

st.set_page_config(page_title="RAG 问答", page_icon="📄", layout="wide")

base_url = st.session_state.get("base_url", api_client.DEFAULT_BASE_URL)

st.title("RAG 问答")
st.info("D12 任务：实现这个页面。参考项目1的 frontend/app.py 的写法。")
raise NotImplementedError("D12 任务：实现「RAG 问答」页面")
