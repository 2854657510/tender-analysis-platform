
import logging

from app.config import settings
from modules.retrieval import bm25_search, vector_store

logger = logging.getLogger(__name__)

# RRF 公式里的平滑常数 k。
# ★ 为什么取 60：这是原论文推荐的默认值，也是业界惯例。
#   它的作用是【压平】头部名次之间的差距 ——
#   k=0 时第 1 名(1.0) 是第 2 名(0.5) 的两倍，太偏激；
#   k=60 时 1/61 ≈ 0.0164、1/62 ≈ 0.0161，差距很小，
#   这样"两路都排第 3"能赢过"一路排第 1、另一路完全没召回"。
RRF_K = 60


def rrf_fuse(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
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
    scores: dict[tuple, float] = {}       # (source, chunk_index) → 累加的 RRF 分数
    docs: dict[tuple, dict] = {}          # (source, chunk_index) → chunk 本身
    hit_by: dict[tuple, list[str]] = {}   # 记一下"这个 chunk 被哪几路召回了"，调试用

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            # ★ 唯一键用 (source, chunk_index)，不能用 text：
            #   两路召回的同一块文本理论上一样，但用文件名+编号更稳妥 ——
            #   编号是入库时分配的，天然唯一；文本一旦被两路做了不同处理就会对不上。
            key = (hit.get("source"), hit.get("chunk_index"))

            if key not in docs:
                # ★ dict.setdefault 的等价写法，但这里要同时初始化三个 dict，
                #   写成显式 if 比 setdefault 更清楚
                docs[key] = dict(hit)          # 复制一份，别改到调用方的数据
                scores[key] = 0.0
                hit_by[key] = []

            # ★ RRF 的核心公式：只看名次，不看分数。
            #   rank 从 1 开始（enumerate 的 start=1），所以是 k + rank 而不是 k + rank - 1
            scores[key] += 1.0 / (k + rank)
            hit_by[key].append(hit.get("retriever", "?"))

    # 组装输出
    fused = []
    for key, doc in docs.items():
        doc["rrf_score"] = scores[key]
        # ★ 把 score 统一成 RRF 分数：下游（rerank / rag）都按 score 降序取，
        #   不统一的话它们拿到的是"向量相似度或 BM25 分数"的混合体，排序会乱。
        doc["score"] = scores[key]
        # 原始分数保留下来，方便对照调试（面试演示时很好用）
        doc["retrievers"] = hit_by[key]
        fused.append(doc)

    fused.sort(key=lambda d: d["rrf_score"], reverse=True)
    return fused


def search(query: str, top_k: int | None = None) -> list[dict]:
    """混合检索主入口：两路并行召回 → RRF 融合。

    实现步骤：
        vec_hits  = vector_store.search(query, top_k=settings.recall_top_k)
        bm25_hits = bm25_search.search(query, top_k=settings.recall_top_k)
        fused = rrf_fuse([vec_hits, bm25_hits])
        return fused[:top_k or settings.recall_top_k]

    ★ 这里两路是【串行】调的。要不要改成线程池并行？
      实测两路加起来通常几十毫秒，而 Python 有 GIL，线程并行的收益
      远不抵多出来的复杂度。等真的成为瓶颈（比如库里有几十万条 chunk）
      再改也不迟 —— 【过早优化是万恶之源】。
    """
    top_k = top_k or settings.recall_top_k

    # ★ 召回阶段用 recall_top_k（默认 20）而不是 top_k：
    #   粗筛就是要多召回一些，把"可能相关"的都捞进来，
    #   后面交给 D5 的 Rerank 去精排。这是"召回率换准确率"的标准做法。
    vec_hits = vector_store.search(query, top_k=settings.recall_top_k)
    bm25_hits = bm25_search.search(query, top_k=settings.recall_top_k)

    fused = rrf_fuse([vec_hits, bm25_hits])

    logger.info(
        "混合检索：向量 %s 条 + BM25 %s 条 → 融合 %s 条",
        len(vec_hits), len(bm25_hits), len(fused),
    )
    return fused[:top_k]
