"""上传 PDF 并入库 —— D12 任务。"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import api_client  # noqa: E402

st.set_page_config(page_title="上传 PDF 并入库", page_icon="📄", layout="wide")

base_url = st.session_state.get("base_url", api_client.DEFAULT_BASE_URL)

st.title("📄 上传 PDF 并入库")
st.caption("上传后系统会自动：解析 → 切分 → 向量化 → 写入知识库")

# ---------------------------------------------------------------------------
# ① 上传区
# ---------------------------------------------------------------------------
uploaded = st.file_uploader("选择 PDF 文件", type=["pdf"])

if uploaded is not None:
    st.info(f"已选择：**{uploaded.name}**（{len(uploaded.getvalue()) / 1024:.0f} KB）")

    if st.button("开始上传", type="primary"):
        # ★ 用 spinner 而不是进度条：后端是"一次请求处理完"的同步接口，
        #   没有中间进度可报。给用户一句"说明正在干什么"的文案，
        #   比一个卡着不动的进度条诚实得多。
        with st.spinner("正在解析、切分、向量化……（首次运行要下载 Embedding 模型，可能较慢）"):
            try:
                result = api_client.upload_document(
                    base_url, uploaded.getvalue(), uploaded.name
                )
                st.success(
                    f"✅ 入库成功：**{result['page_count']} 页** → "
                    f"**{result['chunk_count']} 个 chunk**"
                )
                # ★ 入完库要清缓存：文档列表要立刻显示新文档，
                #   检索相关的缓存也可能过期了。
                st.cache_data.clear()
            except api_client.BackendDown:
                st.error("🔴 连不上后端。请先启动：`uvicorn app.main:app --reload`")
            except api_client.BackendError as e:
                # ★ 把后端的具体原因显示出来。"处理失败：解析结果为空（可能是扫描件）"
                #   比"上传失败"有用一百倍。
                st.error(f"❌ 上传失败（HTTP {e.status_code}）：{e.detail}")

st.divider()

# ---------------------------------------------------------------------------
# ② 已入库文档列表
# ---------------------------------------------------------------------------
st.subheader("已入库文档")


@st.cache_data(ttl=10)
def _load_documents(url: str) -> list[dict]:
    """拉文档列表。

    ★ 加 ttl=10 的缓存：Streamlit 每次交互都会重跑整个脚本，
      不加缓存的话每点一次按钮就打一次后端 —— 又慢又没必要。
      10 秒是"新鲜度"和"请求量"之间的折中。
    """
    return api_client.list_documents(url)


try:
    documents = _load_documents(base_url)
except api_client.BackendDown:
    st.warning("连不上后端，无法获取文档列表。")
    documents = []
except api_client.BackendError as e:
    st.warning(f"获取文档列表失败：{e.detail}")
    documents = []

if not documents:
    st.caption("还没有文档。上传一份试试 👆")
else:
    for doc in documents:
        cols = st.columns([4, 1, 1, 1, 1])

        # 状态用 emoji 一眼区分：成功 ✅ / 失败 ❌ / 处理中 ⏳
        status_icon = {"done": "✅", "failed": "❌"}.get(doc["status"], "⏳")

        cols[0].markdown(f"{status_icon} **{doc['file_name']}**")
        cols[1].caption(f"{doc['page_count']} 页")
        cols[2].caption(f"{doc['chunk_count']} 块")
        # ★ 上传时间只显示到分钟，秒对人没意义，还占地方
        cols[3].caption(f"{str(doc['created_at'])[:16]}")

        # 失败的行把原因显示出来 —— 用户才知道该怎么处理（比如"这是扫描件"）
        if doc["status"] == "failed" and doc.get("error"):
            st.caption(f"　　⚠️ {doc['error']}")

        if cols[4].button("删除", key=f"del_{doc['file_name']}"):
            try:
                res = api_client.delete_document(base_url, doc["file_name"])
                st.success(f"已删除，清掉 {res['deleted_chunks']} 条 chunk")
                # ★ 删完必须清缓存，否则列表还是旧的（要等 10 秒 ttl 才更新）
                st.cache_data.clear()
                st.rerun()
            except api_client.BackendError as e:
                st.error(f"删除失败：{e.detail}")
