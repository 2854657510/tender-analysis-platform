"""RAG 问答 —— D12 任务。"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import api_client  # noqa: E402

st.set_page_config(page_title="RAG 问答", page_icon="💬", layout="wide")

base_url = st.session_state.get("base_url", api_client.DEFAULT_BASE_URL)

st.title("💬 知识库问答")
st.caption("基于已入库的标书回答，答案带出处，可点开看原文")

# ---------------------------------------------------------------------------
# 会话历史
#
# ★ 存进 st.session_state：Streamlit 每次交互都会【从头重跑整个脚本】，
#   普通变量跑完就没了，聊天记录会瞬间清空。
#   session_state 是每个用户会话独立的持久字典，聊天场景必须用它。
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # [{"role": "user"/"assistant", "content":..., "sources":[...]}]

# 重复渲染历史消息（用户往上翻能看到之前的对话）
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 出处只在助手消息后显示
        if msg.get("sources"):
            with st.expander(f"📎 参考资料（{len(msg['sources'])} 条）"):
                for s in msg["sources"]:
                    # ★ 出处格式：[编号] 文件名 第 N 页 —— 和回答正文里的 [1][2] 对得上。
                    #   这是 RAG 应用的标配：让用户能自己核实答案，而不是盲信模型。
                    st.markdown(f"**[{s['index']}]** `{s['source']}` 第 **{s['page']}** 页")
                    st.caption(s["text"] + "…")

# ---------------------------------------------------------------------------
# 输入区
# ---------------------------------------------------------------------------
question = st.chat_input("问点什么，比如：这个项目的预算金额是多少？")

if question:
    # ① 立刻把用户消息渲染出来（不等后端），体感更快
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # ② 调后端
    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            try:
                result = api_client.query(base_url, question)

                st.markdown(result["answer"])

                if result.get("sources"):
                    with st.expander(f"📎 参考资料（{len(result['sources'])} 条）"):
                        for s in result["sources"]:
                            st.markdown(f"**[{s['index']}]** `{s['source']}` 第 **{s['page']}** 页")
                            st.caption(s["text"] + "…")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                    }
                )

            except api_client.BackendDown:
                st.error("🔴 连不上后端，请先启动：`uvicorn app.main:app --reload`")
            except api_client.BackendError as e:
                st.error(f"❌ 出错了（HTTP {e.status_code}）：{e.detail}")

# 侧边栏放一个清空按钮 —— 聊天记录越攒越长，重跑脚本越来越慢
with st.sidebar:
    if st.button("🧹 清空对话"):
        st.session_state.chat_history = []
        st.rerun()
