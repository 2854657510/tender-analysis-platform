import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import db
from app.config import settings
from modules.parsing import chunker, pdf_parser
from modules.retrieval import vector_store

logger = logging.getLogger(__name__)

router = APIRouter()

# 单份 PDF 允许的最大体积（MB）。
MAX_PDF_MB = 50


def _process_pdf(path: Path) -> dict:
   # 解析 + 切分 + 入库（同步阻塞版）。

    source = path.name

    pages = pdf_parser.parse_pdf(path)
    if not pages:
        raise ValueError("解析结果为空：可能是扫描件（图片型 PDF），需要走 OCR")

    chunks = chunker.split_pages(pages)
    vector_store.delete_by_source(source)
    vector_store.add_chunks(chunks, source=source)

    db.upsert_document(source, len(pages), len(chunks), status="done")

    return {"file_name": source, "page_count": len(pages), "chunk_count": len(chunks)}


@router.post("/documents/upload", tags=["documents"])
async def upload_document(file: UploadFile = File(...)):
   # 上传 PDF 并完成解析入库。
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    # 取纯文件名，防路径穿越。
    safe_name = Path(filename).name
    if not safe_name:
        raise HTTPException(status_code=400, detail="文件名不合法")

    content = await file.read()

    # ② 体积校验
    if len(content) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件超过 {MAX_PDF_MB}MB 上限")
    if not content:
        raise HTTPException(status_code=400, detail="文件是空的")

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / safe_name
    dest.write_bytes(content)

    # ③ 丢进线程池执行。
    try:
        result = await run_in_threadpool(_process_pdf, dest)
    except Exception as e:
        logger.exception("上传处理失败：%s", safe_name)
        # ★ 失败也要留痕：前端列表里能显示"failed + 原因"，
        db.upsert_document(safe_name, 0, 0, status="failed", error=str(e)[:500])
        raise HTTPException(status_code=500, detail=f"处理失败：{e}")

    return result


@router.get("/documents", tags=["documents"])
async def list_documents():
    """列出全部文档。这个【可以直接用】—— 已经调好了 db.list_documents()。"""
    from app.db import list_documents as _list

    return {"documents": _list()}


@router.delete("/documents/{file_name}", tags=["documents"])
async def delete_document(file_name: str):
    # 删除文档：向量库 + SQLite 都要清。
    # ★ 同样要防路径穿越：这个参数直接来自 URL，用户可以随便构造
    safe_name = Path(file_name).name

    # ① 先删向量库再删 SQLite 记录。
    deleted_chunks = await run_in_threadpool(vector_store.delete_by_source, safe_name)

    # ② 再删 SQLite 记录
    db.delete_document(safe_name)

    # ③ 顺手删掉磁盘上的原始文件。missing_ok=True：
    try:
        (settings.data_dir / safe_name).unlink(missing_ok=True)
    except OSError as e:
        logger.warning("原始文件删除失败（不影响检索）：%s", e)

    logger.info("已删除文档 %s（清掉 %s 条 chunk）", safe_name, deleted_chunks)
    return {"file_name": safe_name, "deleted_chunks": deleted_chunks}
