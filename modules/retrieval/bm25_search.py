"""BM25 关键词检索 —— D4 任务。

【要做什么】用关键词匹配的方式检索，和向量检索形成互补。

【为什么需要它】纯向量检索有个致命弱点：对【专有名词、编号、精确数字】不敏感。
    比如搜 "XZ-2026-0912"，向量检索可能召回一堆"看起来像编号"的内容，
    因为它在语义空间里把编号当成噪声。
    BM25 按词频匹配，正好补上这块。

【依赖】pip install rank_bm25 jieba
    jieba 用来给中文分词 —— BM25 是按"词"匹配的，中文不切词就等于没切。
"""
from app.config import settings


def build_index(chunks: list[dict]) -> None:
    """对全部 chunk 建 BM25 索引。

    实现步骤：
        1. import jieba
        2. 把每个 chunk 的 text 用 jieba 切成词列表
        3. rank_bm25.BM25Okapi(切好的词列表)
        4. 把索引和 chunks 存到模块级变量（或缓存起来，别每次检索都重建）

    ★ 性能提示：建索引是 O(全量 chunk)，每次检索都重建会很慢。
      用一个全局变量缓存，或者加个"索引是否已建"的标志。
    """
    raise NotImplementedError("D4 任务：实现 BM25 索引构建")


def search(query: str, top_k: int | None = None) -> list[dict]:
    """BM25 检索。

    返回格式必须和 vector_store.search() 一致：
        [{"text": ..., "page": ..., "source": ..., "score": ..., "retriever": "bm25"}, ...]

    ★ 注意 score 的量纲：BM25 的分数可以是任意正数（不像向量相似度是 0-1）。
      【所以后面融合时不能直接比分数】—— 这就是 D4 要用 RRF 的原因。

    D4 验收：
        · 单独跑 BM25 能召回相关结果
        · 搜一个【招标编号】，BM25 能精准命中，而向量检索会漏 —— 记下这个对比
    """
    raise NotImplementedError("D4 任务：实现 BM25 检索")
