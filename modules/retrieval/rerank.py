"""Rerank 重排 —— D5 任务 ★ 效果提升最明显的一步。

【要做什么】召回 20 条 → 精排出最相关的 5 条。

【为什么需要】这是"召回率"和"准确率"的权衡：
    向量检索是【粗筛】：快，但只保证"相关的在里面"，不保证"排在最前面"
    Rerank 是【精排】：慢，但能准确判断"问题 和 文档"的深层相关度
    → 所以流程是"粗筛多召回一些，再精排挑出最好的"

【两种实现，按资源选】
    A. 本地模型：BAAI/bge-reranker-base（效果好，但要下模型、占内存）
    B. API 方案：用大模型给"问题-文档"对打分（简单，但慢且费 token）

    ★ 先用 B 跑通，再考虑换 A。别一上来就卡在装模型上。
"""
from app.config import settings


def rerank(query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
    """对召回结果重新排序。

    输入：hybrid.search() 的输出（20 条左右）
    输出：精排后的 top_k 条，每项加一个 "rerank_score" 字段

    实现提示（API 方案）：
        对每个 hit，构造 prompt 让模型打 0-10 分：
            "问题：{query}\n文档：{hit['text']}\n这段文档和问题的相关度打几分（0-10）？只输出数字。"
        然后按分数降序。

    ★ 降级兜底：Rerank 服务挂了【不能整个崩】，要能退回 RRF 的顺序。
      写个 try/except，失败时直接返回原顺序 —— 这是工程细节，面试能讲。

    D5 验收：
        · Rerank 后 Top5 的相关性明显好于 RRF 直接取 Top5
        · Rerank 失败时能降级，不抛异常
    """
    raise NotImplementedError("D5 任务：实现 Rerank 重排")
