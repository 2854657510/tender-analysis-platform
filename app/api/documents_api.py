"""文档接口 —— D12 任务（但 D2 就会开始用到）。

【三个接口】
    POST   /api/documents/upload   上传 PDF 并入库
    GET    /api/documents          列出已上传的文档
    DELETE /api/documents/{name}   删除文档（同时清向量库）

【上传的完整流程】（这是 D2 的核心验收）
    上传文件 → 存到 data/ → parse_pdf 解析 → split_pages 切分
    → add_chunks 写向量库 → upsert_document 记录到 SQLite
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()


@router.post("/documents/upload", tags=["documents"])
async def upload_document(file: UploadFile = File(...)):
    """上传 PDF 并完成解析入库。

    实现步骤：
        1. 校验后缀是 .pdf
        2. 存到 settings.data_dir / file.filename
        3. pages  = pdf_parser.parse_pdf(path)
        4. chunks = chunker.split_pages(pages)
        5. vector_store.delete_by_source(file.filename)   # 先删旧的，防重复
           vector_store.add_chunks(chunks, source=file.filename)
        6. db.upsert_document(file.filename, len(pages), len(chunks), "done")
        7. 返回 {"file_name":..., "page_count":..., "chunk_count":...}

    ★ 异常处理：解析失败时要写 db.upsert_document(..., status="failed", error=str(e))，
      不要把整个接口搞崩 —— 批量上传时一份坏文件不该拖垮全部。

    D12 验收：上传一份 PDF，能看到页数和 chunk 数
    """
    raise NotImplementedError("D12 任务：实现文档上传")


@router.get("/documents", tags=["documents"])
async def list_documents():
    """列出全部文档。这个【可以直接用】—— 已经调好了 db.list_documents()。"""
    from app.db import list_documents as _list

    return {"documents": _list()}


@router.delete("/documents/{file_name}", tags=["documents"])
async def delete_document(file_name: str):
    """删除文档：向量库 + SQLite 都要清。

    ★ 容易漏的地方：只删了 SQLite 记录，向量库里的 chunk 还在，
      检索时还会召回已删除文档的内容 —— 这是经典的"删不干净"bug。
    """
    raise NotImplementedError("D12 任务：实现文档删除")
