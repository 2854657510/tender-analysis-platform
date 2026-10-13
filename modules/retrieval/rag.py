"""最小 RAG 链路 —— D3 任务 ★★★

【这是整个项目的第一个里程碑】
跑通它，你就有了一个能演示的 RAG 系统。后面 D4/D5 都是在它基础上做优化。

【RAG 的标准流程】
    问题 → 检索相关片段 → 拼进 prompt → 生成回答 → 返回答案 + 出处
                        ↑
                  这一步就是 RAG 区别于普通问答的全部
"""
from modules.retrieval import vector_store

ANSWER_PROMPT = "rag_answer"   # 对应 prompts/rag_answer.j2


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
    raise NotImplementedError("D3 任务：实现上下文拼接")


def answer(question: str) -> dict:
    """RAG 主入口：提问，返回带出处的回答。

    返回格式：
        {
            "answer": "根据招标文件，本项目预算为 1850 万元 [1]...",
            "sources": [
                {"index": 1, "page": 3, "source": "xx.pdf", "text": "原文片段..."},
                ...
            ]
        }

    实现步骤：
        1. hits = vector_store.search(question, top_k=settings.rerank_top_k)
        2. context, sources = build_context(hits)
        3. 用 render(ANSWER_PROMPT, context=..., question=...) 渲染提示词
        4. content = chat([{"role": "user", "content": prompt}])
        5. 返回 {"answer": content, "sources": sources}

    ★★ 治幻觉的关键在提示词里，不在代码里。
        必须在 prompts/rag_answer.j2 里写死这条规则：
            "只依据【参考资料】回答。资料里没有的内容，直接说'资料中未提及'，不要推测。"

    D3 验收（三条都要过）：
        · 端到端跑通「提问 → 检索 → 带出处的回答」
        · 回答里有 [1] [2] 这样的引用编号，且能对应到具体页
        · ★ 故意问一个资料里【没有】的问题，模型说"未提及"而不是编造
    """
    raise NotImplementedError("D3 任务：实现 RAG 主链路")


def answer_with_retrieval_debug(question: str) -> dict:
    """带调试信息的版本（可选）：把召回的每条 chunk 也返回，方便看检索质量。

    用途：调 prompt 时如果回答不对，先看是【检索没召回】还是【模型没读懂】。
    这是 RAG 效果归因的第一步。
    """
    raise NotImplementedError("D3 任务（可选）：带检索明细的回答")
