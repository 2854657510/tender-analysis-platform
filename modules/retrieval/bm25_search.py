
import logging
import threading

import jieba
from rank_bm25 import BM25Okapi

from app.config import settings
from modules.retrieval import vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 索引缓存
#
# ★ 为什么要缓存：建索引要把【全量 chunk】切词 + 算 IDF，是 O(全量) 的操作。
#   每次检索都重建的话，库里有几千条 chunk 时每个问题都要等好几秒。
#
# ★ 怎么知道索引过期了：记下建索引那一刻库里的条数。
#   下次检索时比对当前条数，不一样就说明入库/删除了文档 → 重建。
#   这个判断是 O(1) 的（Chroma 的 count 是直接查元数据），开销可以忽略。
# ---------------------------------------------------------------------------
_chunks: list[dict] = []
_bm25: BM25Okapi | None = None
_indexed_count: int = -1
_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    """中文分词。

    ★ BM25 是【按词】统计词频的。中文不切词的话，一整句话就是一个"词"，
      每个 chunk 之间几乎没有任何重合，BM25 直接失效。
      所以 jieba 这一步不是优化，是必需品。

    ★ 顺手丢掉空白 token：切完常会掉出 " "、"\n" 这种碎片，
      留着会污染词表，让 IDF 失真。
    """
    return [word for word in jieba.cut(text) if word.strip()]


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
    global _chunks, _bm25, _indexed_count

    if not chunks:
        _chunks, _bm25, _indexed_count = [], None, 0
        return

    # ① 切词。这一步最慢，所以只在重建时做一次
    tokenized = [_tokenize(c["text"]) for c in chunks]

    # ② ★ 记下建索引时的语料规模，供 _ensure_index 判断是否过期
    _chunks = chunks
    _bm25 = BM25Okapi(tokenized)
    _indexed_count = len(chunks)

    logger.info("BM25 索引已建：%s 个 chunk，词表 %s 个词", len(chunks), len(_bm25.idf))


def _ensure_index() -> None:
    """确保索引存在且不过期，需要就重建。

    ★ 这是"懒加载 + 自动失效"的合体：
      - 第一次检索时自动建（不用调用方记得手动建）
      - 库里条数变了（新文档入库 / 文档被删）自动重建
    """
    current = vector_store.count()

    # 快路径：索引有效且规模没变 → 直接用，加锁都不需要
    if _bm25 is not None and _indexed_count == current:
        return

    with _lock:
        # 双检：等锁的这段时间里可能已经有别的线程重建好了
        if _bm25 is not None and _indexed_count == vector_store.count():
            return
        build_index(vector_store.all_chunks())


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
    top_k = top_k or settings.recall_top_k
    _ensure_index()

    if _bm25 is None or not _chunks:
        logger.warning("BM25 索引为空，库里可能没有文档")
        return []

    scores = _bm25.get_scores(_tokenize(query))

    # ★ 按分数从高到低排全局名次。注意这里不能直接对 scores 排序取值 ——
    #   我们需要知道"第几名对应哪个 chunk"，所以排的是【下标】。
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits = []
    for i in ranked_indices:
        # ★ 只保留分数 > 0 的。BM25 会给所有文档打分，包括一个词都没命中的，
        #   那些是 0 分 —— 放进结果里只会稀释 RRF 的权重（白占一个名次）。
        if scores[i] <= 0:
            continue

        c = _chunks[i]
        hits.append(
            {
                "text": c["text"],
                "page": c["page"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "score": float(scores[i]),      # ← 量纲和向量不同，这是 RRF 存在的理由
                "retriever": "bm25",
            }
        )
        if len(hits) >= top_k:
            break

    return hits
