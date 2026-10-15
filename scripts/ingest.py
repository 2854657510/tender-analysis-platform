
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import db                              # noqa: E402
from app.config import settings                 # noqa: E402
from modules.parsing import chunker, pdf_parser  # noqa: E402
from modules.retrieval import vector_store       # noqa: E402


def ingest_one(pdf_path: Path) -> dict:
    """解析 + 切分 + 入库一份 PDF。

    实现步骤（和 documents_api 的上传流程一样，只是入口不同）：
        1. pages  = pdf_parser.parse_pdf(pdf_path)
        2. chunks = chunker.split_pages(pages)
        3. vector_store.delete_by_source(pdf_path.name)   # 先删旧的
        4. vector_store.add_chunks(chunks, source=pdf_path.name)
        5. db.upsert_document(pdf_path.name, len(pages), len(chunks), "done")
        6. 返回统计信息

    ★ 单份失败不要中断整批 —— try/except 包住，
      失败的记录到 db 里（status="failed"），继续处理下一份。
      这就是项目1里"降级兜底"的同一个思路。
    """
    pdf_path = Path(pdf_path)          # 允许调用方传 str，统一转成 Path
    source = pdf_path.name             # ★ source 用【文件名】，全项目统一口径

    try:
        # ① 解析：PDF → [{"page": 1, "text": "..."}, ...]
        pages = pdf_parser.parse_pdf(pdf_path)
        if not pages:
            # 解析出来是空的 —— 基本就是扫描件。
            # ★ 不抛异常，而是记为 failed。批量跑的时候，一份坏文件
            #   不该让后面的好文件也处理不了。
            db.upsert_document(source, 0, 0, status="failed", error="解析结果为空（可能是扫描件，需要 OCR）")
            return {"file_name": source, "status": "failed", "error": "解析结果为空（可能是扫描件）"}

        # ② 切分：逐页文本 → 带页码的 chunk
        chunks = chunker.split_pages(pages)

        # ③ 先删旧的：同一份文件重复入库时，避免新旧两批 chunk 同时存在
        vector_store.delete_by_source(source)

        # ④ 写入向量库
        vector_store.add_chunks(chunks, source=source)

        # ⑤ 记一笔元数据到 SQLite
        db.upsert_document(source, len(pages), len(chunks), status="done")

        return {
            "file_name": source,
            "status": "done",
            "page_count": len(pages),
            "chunk_count": len(chunks),
        }

    except Exception as e:
        # ★ 兜底：任何一步炸了都记进 db，然后【正常返回】而不是 raise。
        #   这样 for 循环能继续处理下一份文件。
        import logging

        logging.getLogger(__name__).error("入库失败：%s —— %s", source, e)
        db.upsert_document(source, 0, 0, status="failed", error=str(e)[:500])
        return {"file_name": source, "status": "failed", "error": str(e)[:200]}


if __name__ == "__main__":
    pdfs = (
        [Path(sys.argv[1])]
        if len(sys.argv) > 1
        else sorted(settings.data_dir.glob("*.pdf"))
    )
    if not pdfs:
        print(f"data/ 目录下没有 PDF。把标书放进去：{settings.data_dir}")
        raise SystemExit(0)

    for p in pdfs:
        print(f"\n处理：{p.name}")
        result = ingest_one(p)
        print(f"  {result}")
