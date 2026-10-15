
import logging
import threading

from app.config import settings

logger = logging.getLogger(__name__)

# collection 名字。★ 注意 chromadb 1.x 要求名字 3~512 字符，
# 用 "t" 这种单字母会直接报 InvalidArgumentError。
COLLECTION_NAME = "tender"

# ---------------------------------------------------------------------------
# 模块级缓存：客户端、collection、Embedding 模型
#
# ★ 为什么要缓存：Embedding 模型加载一次要好几秒、占几百 MB 内存，
#   每次检索都重新加载的话，第一个问题就得等十秒。
#   缓存在模块级变量里，整个进程只加载一次。
#
# ★ 为什么要加锁：FastAPI 是多线程的，两个请求同时进来可能同时走到
#   "模型还没加载" 这个分支，于是加载两遍（白烧内存）。
#   双检锁（先无锁判断、再加锁判断）是这类懒加载的标准写法。
# ---------------------------------------------------------------------------
_client = None
_collection = None
_model = None
_lock = threading.Lock()


def _get_model():
    """懒加载中文 Embedding 模型（BGE）。

    ★ 为什么用 BAAI/bge-small-zh-v1.5：
      中文场景【必须】用中文模型。all-MiniLM 这类英文模型在中文上
      基本等于随机向量 —— 检索出来的东西牛头不对马嘴。
      small 版本约 95MB，dim=512，CPU 上跑也够快。

    ★ 首次运行会从 HuggingFace 下载模型。国内网络如果下不动（超时/连接被拒），
      在启动前设一个镜像环境变量即可：
          export HF_ENDPOINT=https://hf-mirror.com      # Git Bash
          $env:HF_ENDPOINT="https://hf-mirror.com"      # PowerShell
    """
    global _model
    if _model is None:                       # 第一道检查：无锁，走热路径时最快
        with _lock:                          # 第二道检查：加锁，防止并发重复加载
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("正在加载 Embedding 模型：%s（首次运行需下载）", settings.embedding_model)
                    _model = SentenceTransformer(settings.embedding_model)
                    logger.info("Embedding 模型加载完成")
                except Exception as e:
                    # 加载失败时给出【可操作】的提示，而不是让用户对着一堆英文栈发呆
                    raise RuntimeError(
                        f"Embedding 模型加载失败：{e}\n"
                        f"排查：① 网络能否访问 HuggingFace；"
                        f"② 试试设镜像 export HF_ENDPOINT=https://hf-mirror.com 后重跑。"
                    ) from e
    return _model


def _embed(texts: list[str]) -> list[list[float]]:
    """把文本批量转成向量。

    ★ normalize_embeddings=True 是关键：先归一化，向量的点积就等于余弦相似度。
      配合下面 collection 建的 cosine 空间，相似度才能直接算出来。

    ★ 可选优化（不影响跑通，先不做）：BGE 官方建议【查询】加一句指令前缀
      "为这个句子生成表示以用于检索相关文章："，而【文档】不加。
      加上一般能提升一两个点。这里为了逻辑简单，查询和文档统一编码。
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    # Chroma 要的是纯 Python list，numpy 数组传进去某些版本会报错
    return vectors.tolist()


def _get_collection():
    """拿到 collection（不存在就建）。

    ★ metadata={"hnsw:space": "cosine"}：把距离度量设成【余弦距离】。
      Chroma 默认是 L2（欧氏距离），对归一化过的 BGE 向量来说，
      cosine 的区分度更好，也是 BGE 官方推荐的度量。
      ★ 注意：这项在建 collection 时定死，之后再改无效（要重建 collection）。

    ★ 我们【不使用】Chroma 自带的 embedding_function：
      在 chromadb 1.x 里，不显式指定的话它会挂一个默认的 ONNX MiniLM，
      那是个英文模型，而且第一次用还会偷偷下载。
      我们要自己控制用哪个模型，所以向量一律【在 Python 侧算好再传进去】。
    """
    global _client, _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                import chromadb

                # 确保目录存在（PersistentClient 自己建不了多级目录）
                settings.chroma_dir.mkdir(parents=True, exist_ok=True)
                _client = chromadb.PersistentClient(path=str(settings.chroma_dir))
                _collection = _client.get_or_create_collection(
                    COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("向量库就绪：%s（当前 %s 条）", settings.chroma_dir, _collection.count())
    return _collection


def add_chunks(chunks: list[dict], source: str) -> int:
    """把 chunk 批量写入向量库，返回写入条数。

    输入：chunker.split_pages() 的输出，外加一个 source（文件名，用来区分不同文档）
        [{"text": "...", "page": 1, "chunk_index": 0}, ...]

    ★★ 关键：metadatas 里必须存 page 和 source
        metadatas=[{"page": c["page"], "source": source, "chunk_index": c["chunk_index"]}]
        没有它，检索出来就不知道内容来自哪 —— 回答标不出出处。

    D2 验收：
        · 一份 PDF 的 chunk 能全部写入
        · collection.count() 的数字和 chunk 总数一致
        · 再次运行同一份文件【不应重复写入】（去重：先按 source 删掉旧的）
    """
    if not chunks:
        return 0

    collection = _get_collection()

    # ★ id 必须【全局唯一】。用 "文件名_编号" 拼，同一份文件的同一块永远同一个 id，
    #   这样重复写入时能天然覆盖（幂等），而不是报 DuplicateIDError。
    ids = [f"{source}_{c['chunk_index']}" for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "page": c["page"],                  # ← 页码，回答标出处靠它
            "source": source,                   # ← 文件名，删文档/标出处靠它
            "chunk_index": c["chunk_index"],    # ← 全局编号，RRF 融合时做唯一键的一半
        }
        for c in chunks
    ]
    vectors = _embed(texts)

    # ★ 分批写入。一次塞几千条会撑爆内存、也容易触发 Chroma 内部限制。
    #   100 是个保守值，实测很稳。
    batch = 100
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i : i + batch],
            embeddings=vectors[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )

    logger.info("已写入 %s 条 chunk（来源：%s）", len(ids), source)
    return len(ids)


def search(query: str, top_k: int | None = None) -> list[dict]:
    """向量检索（语义相似）。

    返回格式（全项目统一，后面所有检索函数都返回这个结构）：
        [
            {"text": "...", "page": 1, "source": "xxx.pdf", "score": 0.87, "retriever": "vector"},
            ...
        ]
        ★ score 越大越相关（Chroma 默认返回的是【距离】，要自己转成相似度）

    D2 验收：
        · 用一句问句能检索出相关 chunk
        · 打印结果时能看到 page 和 source
        · ★ 构造一个"纯向量会漏"的例子先记下来，D4 用（比如搜一个招标编号）
    """
    top_k = top_k or settings.rerank_top_k
    collection = _get_collection()

    # 空库直接返回。★ 不判断的话 Chroma 会抛异常，第一次用很容易踩。
    if collection.count() == 0:
        logger.warning("向量库是空的，先上传或跑 scripts/ingest.py 入库")
        return []

    query_vector = _embed([query])

    # ★ n_results 不能超过库里已有的条数，否则 Chroma 报错
    n = min(top_k, collection.count())
    result = collection.query(query_embeddings=query_vector, n_results=n)

    # ★ Chroma 返回的每一项都是【嵌套列表】（因为它支持一次查多个 query）。
    #   我们只查了一个 query，所以取 [0] 把外层剥掉。
    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "page": meta.get("page"),
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index", -1),
                # ★ 距离转相似度。cosine 空间下：距离 = 1 - 余弦相似度，
                #   所以相似度 = 1 - 距离，天然落在 0~1，越大越相关。
                "score": 1.0 - distance,
                "retriever": "vector",
            }
        )
    return hits


def delete_by_source(source: str) -> int:
    """按文件名删除该文档的所有 chunk。重新上传同一份文件时先调它，避免重复。

    返回删掉的条数（用"删前数 - 删后数"算出来）。

    ★ 为什么单独提供这个方法：只删 SQLite 记录、忘了删向量库，
      是 RAG 项目最经典的 "删不干净" bug ——
      界面上文档没了，提问时它还会冒出来当参考资料。
    """
    collection = _get_collection()
    before = collection.count()
    # ★ where 是【按元数据过滤】。必须用 where={"source": ...}，
    #   不能传 ids 之类的 —— 我们不知道这份文件有哪些 id，只知道它是谁。
    collection.delete(where={"source": source})
    deleted = before - collection.count()
    if deleted:
        logger.info("已从向量库删除 %s 条（来源：%s）", deleted, source)
    return deleted


def count() -> int:
    """当前向量库里有多少条 chunk。

    ★ 用途：D4 的 BM25 靠它判断"索引是不是过期了" ——
      入库新文档后 count 会变，索引就得重建（见 bm25_search）。
    """
    return _get_collection().count()


def all_chunks() -> list[dict]:
    """把库里【全部】chunk 取出来，格式和 chunker.split_pages() 的输出对齐。

    ★ 用途：D4 建 BM25 索引需要全量文本 —— BM25 的 IDF 是根据整个语料库
      算出来的，只看一部分数据算出来的权重是错的。

    ★ 为什么要分页取：Chroma 的 get() 一次返回太多会吃内存，
      而且底层有单次返回条数的上限。500 一批很稳。
    """
    collection = _get_collection()
    total = collection.count()
    if total == 0:
        return []

    chunks = []
    batch = 500
    for offset in range(0, total, batch):
        result = collection.get(limit=batch, offset=offset, include=["documents", "metadatas"])
        for text, meta in zip(result["documents"], result["metadatas"]):
            chunks.append(
                {
                    "text": text,
                    "page": meta.get("page"),
                    "source": meta.get("source"),
                    "chunk_index": meta.get("chunk_index", -1),
                }
            )
    return chunks


def reset() -> None:
    """清空整个 collection（调试用）。

    ★ 只在本地调试时用。跑完记得重新 ingest，否则检索会一直返回空。
    """
    global _collection
    client = chromadb_client()
    client.delete_collection(COLLECTION_NAME)
    _collection = None
    logger.warning("已清空向量库 collection：%s", COLLECTION_NAME)


def chromadb_client():
    """暴露底层 client（reset 和调试用）。"""
    _get_collection()      # 保证 _client 已初始化
    return _client
