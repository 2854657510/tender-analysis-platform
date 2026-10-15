
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from modules.extraction import extract as extraction
from modules.llm.client import chat
from modules.retrieval import hybrid
from modules.retrieval.rag import build_context

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    question: str          # 用户问题
    needs_retrieval: bool  # 意图识别的结果
    hits: list             # 检索到的片段
    answer: str            # 最终回答
    extracted: dict


# 意图识别用的提示词 分流
INTENT_PROMPT = """判断下面的问题是否需要在【招投标文档库】中检索资料才能回答。

需要检索的：问某个项目的预算、截止时间、资格要求、采购人、联系方式等具体事实。
不需要检索的：打招呼、问你能做什么、纯闲聊、或者通用知识问题。
只回答一个字："要" 或 "不要"，不要任何解释。
问题：{question}"""

# 不走检索时的回答提示词
DIRECT_PROMPT = """你是一个招投标文档分析助手。
用户的问题不需要查询文档库，请直接、简洁地回答。
如果不确定或涉及具体项目数据，请提示用户提供更具体的问题，以便从文档中检索。
用户问题：{question}"""


def node_classify_intent(state: AgentState) -> dict:
   # 意图识别 —— 判断这个问题需不需要检索文档。

    question = state["question"]
    try:
        verdict = chat([{"role": "user", "content": INTENT_PROMPT.format(question=question)}],
                       max_tokens=8).strip()
        needs = "要" in verdict and "不要" not in verdict
        logger.info("意图识别：%s → 需要检索=%s", question[:30], needs)
    except Exception as e:
        # 降级
        logger.warning("意图识别失败，默认走检索：%s", e)
        needs = True
    return {"needs_retrieval": needs}


def node_retrieve(state: AgentState) -> dict:
    # 检索 —— 调 hybrid.search()
    hits = hybrid.search(state["question"], top_k=5)
    logger.info("检索节点：召回 %s 条", len(hits))
    return {"hits": hits}


def node_extract(state: AgentState) -> dict:
    # 抽取
    hits = state.get("hits") or []
    if not hits:
        return {"extracted": {}}
    merged = "\n".join(h["text"] for h in hits[:3])
    # 降级
    try:
        return {"extracted": extraction.extract_safe(merged)}
    except Exception as e:
        logger.warning("抽取节点失败，跳过：%s", e)
        return {"extracted": {}}


def node_generate(state: AgentState) -> dict:
    # 把检索结果和问题拼成 prompt，调模型
    question = state["question"]
    hits = state.get("hits") or []

    if hits:
        # 有资料：走标准 RAG 路径，答案必须带 [1][2] 这样的出处编号
        context, _ = build_context(hits)

        # ★ 把抽取到的结构化字段作为【额外线索】附在问题后面。
        #   注意措辞是"可能相关，需以资料为准"——
        extracted = state.get("extracted") or {}
        hint = ""
        if extracted and not extracted.get("_failed"):
            keys = ("project_name", "buyer", "budget_wan", "deadline")
            useful = {k: extracted[k] for k in keys if extracted.get(k)}
            if useful:
                hint = f"\n\n（系统预抽取的字段，仅供参考，回答请以参考资料原文为准：{useful}）"

        prompt = (
            "你是一位资深的招投标文件分析助手。\n\n"
            "只依据下面的【参考资料】回答，资料中没有的内容直接说\"资料中未提及\"，不要编造。\n"
            "引用资料时标注编号，格式如 [1]、[2]。\n\n"
            f"参考资料：\n{context}\n\n"
            f"用户问题：{question}{hint}\n\n回答："
        )
    else:
        # 无资料：直接回答
        prompt = DIRECT_PROMPT.format(question=question)

    answer = chat([{"role": "user", "content": prompt}])
    return {"answer": answer}


def route_after_classify(state: AgentState) -> str:
    """条件边：意图识别之后走哪条路。

    返回值必须和 add_conditional_edges 里注册的键名【完全一致】。
      写成 "retrieve "（多个空格）或者 "Retrieve"（大写），
    """
    return "retrieve" if state.get("needs_retrieval") else "direct"


def build_graph():
    """组装状态图。

    结构：
        START → classify → (条件边) ─┬→ retrieve → extract ─┐
                                     └→ generate ────────────┴→ generate → END


    """
    g = StateGraph(AgentState)

    # ① 注册节点
    g.add_node("classify", node_classify_intent)
    g.add_node("retrieve", node_retrieve)
    g.add_node("extract", node_extract)
    g.add_node("generate", node_generate)

    # ② 入口：从 classify 开始
    g.set_entry_point("classify")

    # ③ 条件边。★ 第二个参数是"判断函数"，第三个是"返回值 → 节点名"的映射表
    #    route_after_classify 返回 "retrieve" 就跳到 retrieve 节点，
    #    返回 "direct" 就跳到 generate —— 键名必须和函数返回值一模一样。
    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {"retrieve": "retrieve", "direct": "generate"},
    )

    # ④ 固定边：检索 → 抽取 → 生成，最后结束
    g.add_edge("retrieve", "extract")
    g.add_edge("extract", "generate")
    g.add_edge("generate", END)

    return g.compile()


if __name__ == "__main__":
    print(build_graph().get_graph().draw_mermaid())
