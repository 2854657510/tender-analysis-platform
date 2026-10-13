"""混合检索 —— D4 任务 ★

【要做什么】把向量检索和 BM25 的结果融合成一个排名。

【核心问题】两路结果的【分数没法直接比】：
    向量相似度是 0-1 之间的小数，BM25 分数可能是 0-30。
    直接加权求和 = 拿苹果加橘子。

【解法：RRF（Reciprocal Rank Fusion）】
    不看分数，【只看排名】。
    每个文档的得分 = Σ 1 / (k + rank)，k 通常取 60。
    排第 1 的贡献 1/61，排第 2 的贡献 1/62……

    ★ 为什么这样有效：它把"两路都觉得不错的文档"顶上来，
      同时天然消除了量纲差异 —— 这是面试重点。
"""
from modules.retrieval import bm25_search, vector_store


def rrf_fuse(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """RRF 融合多路检索结果。

    输入：[vector_results, bm25_results]
    输出：融合后按 RRF 分数降序的列表，每项带 "rrf_score" 字段

    实现步骤：
        1. 用一个 dict 按 (source, chunk_index) 聚合同一个 chunk
           —— 两路可能召回同一个 chunk，要合并而不是重复
        2. 遍历每一路，对排名第 i 的结果累加 1 / (k + i + 1)
        3. 按累加分数排序输出

    D4 验收：
        · 融合结果里能看到同一个 chunk 被两路都召回（rrf_score 明显更高）
        · 融合后的 Top5 质量比单路好
    """
    raise NotImplementedError("D4 任务：实现 RRF 融合")


def search(query: str, top_k: int | None = None) -> list[dict]:
    """混合检索主入口：两路并行召回 → RRF 融合。

    实现步骤：
        vec_hits  = vector_store.search(query, top_k=settings.recall_top_k)
        bm25_hits = bm25_search.search(query, top_k=settings.recall_top_k)
        fused = rrf_fuse([vec_hits, bm25_hits])
        return fused[:top_k or settings.recall_top_k]
    """
    raise NotImplementedError("D4 任务：实现混合检索主入口")
