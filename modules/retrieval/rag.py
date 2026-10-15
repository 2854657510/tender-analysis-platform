
import logging

from app.config import settings
from modules.extraction.prompt import render
from modules.llm.client import chat
from modules.retrieval import hybrid, query_rewrite, rerank, vector_store

logger = logging.getLogger(__name__)

ANSWER_PROMPT = "rag_answer"   # 对应 prompts/rag_answer.j2

# 单条出处摘要回传给前端时的最大长度。
# ★ 前端只需要展示"这一条大概讲了什么"，回传全文会让响应体膨胀好几倍
#   （20 条 × 上千字），而且前端本来就有跳回原文的链接。
SOURCE_SNIPPET_CHARS = 200


def build_context(hits: list[dict]) -> tuple[str, list[dict]]:
    """把检索结果拼成给模型看的上下文。

    输入：search() 返回的列表
    输出：(拼好的上下文字符串, 出处列表)

    格式要求 —— 每条资料要带【编号】和【出处】，方便模型引用：
        [1]（来源：xx.pdf 第 3 页）
        招标人：XX市交通运输局...

        [2]（来源：xx.pdf 第 7 页）
        预算金额：1850 万元...

    ★ 为什么要编号：让模型能写 "根据 [1]" 这样的引用，
      前端就能把 [1] 变成可点击的链接，跳回原文。这是 RAG 应用的标配交互。
    """
    blocks = []
    sources = []

    for i, hit in enumerate(hits, start=1):
        # ★ 编号【从 1 开始】，和提示词里 "标注编号，格式如 [1]、[2]" 的约定对齐。
        #   如果用默认的 0 起，模型会写出 [0]，前端按 1 起找链接就全错位了。
        header = f"[{i}]（来源：{hit.get('source')} 第 {hit.get('page')} 页）"
        blocks.append(f"{header}\n{hit['text']}")

        # ★ 出处列表单独返回给前端 —— 模型只负责在正文里写 [1]，
        #   "1 到底对应哪一页"这件事由结构化数据保证，不靠模型转述（它会说错）。
        sources.append(
            {
                "index": i,
                "page": hit.get("page"),
                "source": hit.get("source"),
                "text": hit["text"][:SOURCE_SNIPPET_CHARS],
            }
        )

    # ★ 块之间用空行分隔：Jinja 模板里 context 是整段插进去的，
    #   挨在一起会让模型分不清 [1] 的内容到哪里结束。
    context = "\n\n".join(blocks)
    return context, sources


def _retrieve(question: str, use_rewrite: bool = False, top_k: int | None = None) -> list[dict]:
    """检索流程：改写（可选）→ 混合召回 → 精排。

    ★ 这是 D3~D5 三层优化叠在一起的结果：
      D3 只有向量检索，D4 加了 BM25 + RRF，D5 加了 Rerank。
      这里直接给出【最终形态】，注释里标了每一步是哪天加的。

    ★ 为什么 use_rewrite 默认关闭：改写要多花一次模型调用。
      它只在"用户问法很口语、检索老是漏"时才划算；
      默认开着等于每次问答都多花钱，换来的提升不稳定。
    """
    query = question
    if use_rewrite:
        query = query_rewrite.rewrite(question)      # D5

    # D4：混合召回（内部已经是 向量 + BM25 → RRF）
    candidates = hybrid.search(query, top_k=settings.recall_top_k)
    if not candidates:
        return []

    # D5：精排。★ rerank 内部有降级兜底，失败时返回原顺序，不会抛异常
    return rerank.rerank(query, candidates, top_k=top_k or settings.rerank_top_k)


def answer(question: str, use_rewrite: bool = False, top_k: int | None = None) -> dict:
    """RAG 主入口：提问，返回带出处的回答。

    返回格式：
        {
            "answer": "根据招标文件，本项目预算为 1850 万元 [1]...",
            "sources": [
                {"index": 1, "page": 3, "source": "xx.pdf", "text": "原文片段..."},
                ...
            ]
        }

    ★★ 治幻觉的关键在提示词里，不在代码里。
        必须在 prompts/rag_answer.j2 里写死这条规则：
            "只依据【参考资料】回答。资料里没有的内容，直接说'资料中未提及'，不要推测。"

    D3 验收（三条都要过）：
        · 端到端跑通「提问 → 检索 → 带出处的回答」
        · 回答里有 [1] [2] 这样的引用编号，且能对应到具体页
        · ★ 故意问一个资料里【没有】的问题，模型说"未提及"而不是编造
    """
    hits = _retrieve(question, use_rewrite=use_rewrite, top_k=top_k)

    # ★ 一条都没召回时【不要调模型】。
    #   没有参考资料还硬问，模型只能靠自己的先验知识编 —— 那就不是 RAG 了。
    #   直接返回固定话术，既省钱又杜绝幻觉。
    if not hits:
        return {
            "answer": "知识库里没有检索到相关内容。请先上传招投标文档，或换一种问法。",
            "sources": [],
        }

    context, sources = build_context(hits)

    # ★ 拼提示词：模板里已经写死了"只依据参考资料回答"的规则，
    #   我们只负责把 context 和 question 填进去。
    prompt = render(ANSWER_PROMPT, context=context, question=question)

    content = chat([{"role": "user", "content": prompt}])

    return {"answer": content, "sources": sources}


def answer_with_retrieval_debug(question: str, use_rewrite: bool = False) -> dict:
    """带调试信息的版本（可选）：把召回的每条 chunk 也返回，方便看检索质量。

    用途：调 prompt 时如果回答不对，先看是【检索没召回】还是【模型没读懂】。
    这是 RAG 效果归因的第一步。
    """
    hits = _retrieve(question, use_rewrite=use_rewrite)

    if not hits:
        return {"answer": "知识库里没有检索到相关内容。", "sources": [], "retrieval": []}

    context, sources = build_context(hits)
    prompt = render(ANSWER_PROMPT, context=context, question=question)
    content = chat([{"role": "user", "content": prompt}])

    # ★ 把检索明细原样带出去 —— 每条都标注了它是走哪条路被召回的
    #   （retrievers 里有 vector / bm25），这是调参时最有用的信息：
    #   如果正确答案根本没进候选，那是召回的问题，改 prompt 一万遍也没用。
    retrieval = [
        {
            "index": i,
            "source": h.get("source"),
            "page": h.get("page"),
            "score": round(h.get("score", 0), 4),
            "rerank_score": h.get("rerank_score"),
            "retrievers": h.get("retrievers", [h.get("retriever")]),
            "text": h["text"][:120],
        }
        for i, h in enumerate(hits, start=1)
    ]

    return {"answer": content, "sources": sources, "retrieval": retrieval}
