"""向量库封装 —— D2 任务。

【要做什么】把 chunk 存进 Chroma，并能按语义检索出来。

【依赖】pip install chromadb

【面试会问的选型问题】为什么用 Chroma 不用 Milvus？
    答：数据量没到那个级别。Chroma 能嵌进进程、零运维、开发期最快；
        Milvus 是为亿级向量设计的，要独立部署，百万级以下属于过度设计。
        **选型的依据是数据规模和运维成本，不是"哪个更牛"。**
"""
from app.config import settings


def add_chunks(chunks: list[dict], source: str) -> int:
    """把 chunk 批量写入向量库，返回写入条数。

    输入：chunker.split_pages() 的输出，外加一个 source（文件名，用来区分不同文档）
        [{"text": "...", "page": 1, "chunk_index": 0}, ...]

    ★★ 关键：metadatas 里必须存 page 和 source
        metadatas=[{"page": c["page"], "source": source, "chunk_index": c["chunk_index"]}]
        没有它，检索出来就不知道内容来自哪 —— 回答标不出出处。

    Chroma 用法提示：
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        collection = client.get_or_create_collection("tender")
        collection.add(documents=[...], metadatas=[...], ids=[...])
        # ids 必须唯一，用 f"{source}_{chunk_index}" 即可

    D2 验收：
        · 一份 PDF 的 chunk 能全部写入
        · collection.count() 的数字和 chunk 总数一致
        · 再次运行同一份文件【不应重复写入】（去重：先按 source 删掉旧的）
    """
    raise NotImplementedError("D2 任务：实现向量库写入")


def search(query: str, top_k: int | None = None) -> list[dict]:
    """向量检索（语义相似）。

    返回格式（全项目统一，后面所有检索函数都返回这个结构）：
        [
            {"text": "...", "page": 1, "source": "xxx.pdf", "score": 0.87, "retriever": "vector"},
            ...
        ]
        ★ score 越大越相关（Chroma 默认返回的是【距离】，要自己转成相似度）

    实现提示：
        results = collection.query(query_texts=[query], n_results=top_k)
        # 返回的是嵌套列表：documents[0]、metadatas[0]、distances[0]

    D2 验收：
        · 用一句问句能检索出相关 chunk
        · 打印结果时能看到 page 和 source
        · ★ 构造一个"纯向量会漏"的例子先记下来，D4 用（比如搜一个招标编号）
    """
    raise NotImplementedError("D2 任务：实现向量检索")


def delete_by_source(source: str) -> int:
    """按文件名删除该文档的所有 chunk。重新上传同一份文件时先调它，避免重复。"""
    raise NotImplementedError("D2 任务：实现按来源删除")
