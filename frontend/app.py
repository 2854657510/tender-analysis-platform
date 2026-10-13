"""首页 —— D12 任务。

启动：
    streamlit run frontend/app.py

【Streamlit 多页面约定】pages/ 目录下的 .py 文件会自动出现在侧边栏导航里，
文件名前缀的数字决定顺序（1_ 在最前）。
"""
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

st.title("📑 智能标书分析平台")
st.caption("上传标书 → 建立知识库 → 自然语言问答 → Agent 自动分析")

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
只能 Ctrl+F 或者从头翻。多家标书对比更是灾难。

**方案**：把标书变成可检索的知识库，用 RAG 做精准问答；
复杂的对比分析交给 Agent 拆解执行。

**和项目1的关系**：项目1做了单篇文档的结构化抽取，
但处理不了"跨文档检索"——这就是本项目引入 RAG 的原因。
""")
