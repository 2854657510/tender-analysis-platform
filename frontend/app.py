
import streamlit as st

import api_client

st.set_page_config(page_title="智能标书分析平台", page_icon="📑", layout="wide")

with st.sidebar:
    st.header("⚙️ 后端连接")
    base_url = st.text_input(
        "后端地址", value=api_client.DEFAULT_BASE_URL, help="先启动后端服务"
    )
    if api_client.check_health(base_url):
        st.success("🟢 后端已连接", icon="✅")
    else:
        st.error("🔴 连不上后端")
        st.caption("另开一个终端运行：\n`uvicorn app.main:app --reload`")

    # ★★ 存进 session_state，pages/ 下的三个子页面才能拿到【用户填的这个地址】。
    #    子页面里写的是 st.session_state.get("base_url", 默认值) ——
    #    不存的话它们永远只会用默认的 127.0.0.1:8000，
    #    用户在后端部署到服务器后就怎么都连不上了。
    st.session_state["base_url"] = base_url

st.title("📑 智能标书分析平台")
st.caption("基于 RAG 技术，帮助用户快速分析和对比多个标书。")

st.markdown("""
### 怎么用

| 页面 | 干什么 |
|---|---|
| **1 上传文档** | 拖入 PDF，系统自动解析、切分、入库 |
| **2 知识库问答** | 用自然语言提问，得到带出处的回答 |
| **3 报告分析** | 复杂问题交给 Agent，自动拆解成多步完成 |

👈 从左侧导航开始。

### 这个项目在做什么

**痛点**：一份标书几十上百页，想找"这个项目的资格要求是什么"，
只能 Ctrl+F 或者从头翻。

**方案**：把标书变成可检索的知识库，用 RAG 做精准问答；
复杂的对比分析交给 Agent 拆解执行。
""")
