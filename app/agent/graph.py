"""LangGraph 状态图编排 —— D7 任务。

【要做什么】把 D6 手写的循环，改造成一张能画出来的状态图。

【和 D6 手写循环的区别】
    D6：模型自由决定下一步 → 灵活但不可控
    D7：流程有明确节点和条件边 → 可控、可画、可维护
    ★ 面试会问"什么时候用哪个"。答：流程固定用图，开放式探索用循环。

【三个核心概念】
    State              在节点间传递的字典
    Node               一个函数，收 State、返回 State 的更新
    Conditional Edge   根据 State 的某个值决定走哪条分支

【依赖】pip install langgraph
"""
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """在节点之间传递的状态。

    ★ 每个字段都要标类型 —— LangGraph 靠类型决定怎么合并字段。
    ★ 需要【追加】而不是【覆盖】的字段，用 Annotated + operator.add，
      否则后一个节点会把前一个节点的结果冲掉。
    """

    question: str          # 用户问题
    needs_retrieval: bool  # 意图识别的结果
    hits: list             # 检索到的片段
    answer: str            # 最终回答


def node_classify_intent(state: AgentState) -> dict:
    """节点1：意图识别 —— 判断这个问题需不需要检索文档。

    返回 {"needs_retrieval": True/False}（只返回要更新的字段，不用返回整个 state）
    """
    raise NotImplementedError("D7 任务：实现意图识别节点")


def node_retrieve(state: AgentState) -> dict:
    """节点2：检索 —— 调 hybrid.search()"""
    raise NotImplementedError("D7 任务：实现检索节点")


def node_extract(state: AgentState) -> dict:
    """节点3：抽取 —— 复用项目1的能力"""
    raise NotImplementedError("D7 任务：实现抽取节点")


def node_generate(state: AgentState) -> dict:
    """节点4：生成回答 —— 把检索结果和问题拼成 prompt，调模型"""
    raise NotImplementedError("D7 任务：实现生成节点")


def route_after_classify(state: AgentState) -> str:
    """条件边：意图识别之后走哪条路。

    ★ 返回值必须和 add_conditional_edges 里注册的键名【完全一致】。
    """
    raise NotImplementedError("D7 任务：实现条件路由")


def build_graph():
    """组装状态图。

    结构：
        START → classify → (条件边) ─┬→ retrieve → extract ─┐
                                     └→ generate ────────────┴→ generate → END

    实现步骤：
        g = StateGraph(AgentState)
        g.add_node("classify", node_classify_intent)
        ... 把节点都加上
        g.set_entry_point("classify")
        g.add_conditional_edges("classify", route_after_classify,
                                {"retrieve": "retrieve", "direct": "generate"})
        g.add_edge("retrieve", "extract")
        g.add_edge("extract", "generate")
        g.add_edge("generate", END)
        return g.compile()

    ★ 编译后建议打印一下图：print(app.get_graph().draw_mermaid())
      可以直接贴进 GitHub README，面试时也能展示。

    D7 验收：
        · 状态机跑通，"需要检索"和"不需要检索"两种情况都走对分支
        · ★ 能手画出这张状态图
        · 能说出"什么时候不该用 LangGraph"
    """
    raise NotImplementedError("D7 任务：组装状态图")
