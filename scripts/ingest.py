"""批量入库脚本 —— D2 任务。

【用途】把 data/ 目录下的所有 PDF 一次性解析入库。
       比在网页上一个个上传快得多，调试时很有用。

用法：
    python scripts/ingest.py              # 处理 data/ 下所有 PDF
    python scripts/ingest.py 某文件.pdf    # 只处理指定文件
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings   # noqa: E402


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
    raise NotImplementedError("D2 任务：实现单份入库")


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
