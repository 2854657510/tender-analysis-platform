"""Agent 分析 —— D12 任务。"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import api_client  # noqa: E402

st.set_page_config(page_title="Agent 分析", page_icon="🤖", layout="wide")

base_url = st.session_state.get("base_url", api_client.DEFAULT_BASE_URL)

st.title("🤖 Agent 分析")
st.caption("复杂问题交给 Agent —— 它自己决定调哪些工具、分几步完成")

st.markdown("""
**和「知识库问答」的区别**：
- 问答页：一次检索 → 一次回答，适合"某某是多少"这类**单点事实**问题
- 本页：Agent 自己规划步骤、反复调工具，适合"**对比几个项目**"、"**汇总多处信息**"这类需要多步的问题
""")

# ---------------------------------------------------------------------------
# 预设几个典型问题，一点就跑 —— 演示时不用现场打字，也方便面试官自己试
# ---------------------------------------------------------------------------
EXAMPLES = [
    "这三个项目的预算分别是多少？哪个最高？",
    "对比这几份标书的资格要求差异",
    "XX市智慧交通建设项目的截止时间是什么时候？",
]

st.markdown("**试试这些：**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.pending_question = example

question = st.text_area(
    "分析需求",
    value=st.session_state.get("pending_question", ""),
    height=80,
    placeholder="例：对比这几个项目的预算和资格要求",
)

if st.button("开始分析", type="primary") and question.strip():
    # ★ timeout 给足：Agent 要跑好几轮模型调用，每轮几秒，
    #   总共十几二十秒很正常，前端不能提前掐断。
    with st.spinner("Agent 正在思考、调用工具……"):
        try:
            result = api_client.analyze(base_url, question)
        except api_client.BackendDown:
            st.error("🔴 连不上后端，请先启动：`uvicorn app.main:app --reload`")
            st.stop()
        except api_client.BackendError as e:
            st.error(f"❌ 分析失败（HTTP {e.status_code}）：{e.detail}")
            st.stop()

    # -----------------------------------------------------------------------
    # ★ 先展示执行过程，再展示答案
    #
    #   这是本页面的【核心亮点】：把 Agent 的内部动作摊开给用户看。
    #   普通问答只有一个答案，你看不出它到底做了什么；
    #   这里能看到"它先检索了 A，又抽取了 B"—— 这才是 Agent 的价值可视化。
    #   面试时这一屏就是最好的证据。
    # -----------------------------------------------------------------------
    steps = result.get("steps") or []

    if steps:
        with st.expander(f"🔍 Agent 执行过程（{len(steps)} 步）", expanded=True):
            for s in steps:
                if s.get("error"):
                    st.markdown(f"**第 {s['step']} 步** ❌ `{s['tool']}` — {s['error']}")
                else:
                    st.markdown(f"**第 {s['step']} 步** ✅ `{s['tool']}`")
                    st.caption(f"参数：{s.get('args')}")
                    if s.get("result"):
                        # 工具结果可能很长，用 code 块限高显示，别把页面撑爆
                        st.code(str(s["result"])[:400], language="json")
    else:
        st.info("这个问题的回答不需要调用工具。")

    # 停止原因：正常结束还是撞上了步数上限
    if result.get("stopped_reason") == "max_steps":
        st.warning("⚠️ Agent 达到最大步数仍未得出结论，建议把问题拆得更具体一些。")

    st.subheader("📋 分析结论")
    st.markdown(result.get("answer", "（无内容）"))
