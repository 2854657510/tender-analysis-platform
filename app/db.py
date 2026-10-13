"""SQLite 文档表 —— D2 任务。

【为什么需要它】向量库存的是"文本片段"，但你还得知道：
    · 有哪些文档被上传过
    · 每份文档解析成功没有
    · 什么时候传的、多少页

这些"元数据"用关系库存最合适。向量库 + 关系库是 RAG 的标准组合。

【说明】这个文件是【可直接运行】的，你不需要重写。
        D2 的活是把它接进上传流程里（documents_api.py）。
"""
import sqlite3
from datetime import datetime

from app.config import settings

DB_PATH = settings.data_dir / "documents.db"


def get_conn() -> sqlite3.Connection:
    """拿一个数据库连接。

    ★ check_same_thread=False：FastAPI 是多线程的，
      默认 SQLite 连接不允许跨线程用，不加这个参数会报错。
      （生产环境更严谨的做法是用连接池，但 demo 阶段这样够用。）
    """
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # 让查询结果能像字典一样用：row["file_name"]
    return conn


def init_db() -> None:
    """建表。应用启动时调一次，重复调用不会出错（IF NOT EXISTS）。"""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name   TEXT NOT NULL UNIQUE,   -- 文件名唯一，重复上传时走更新
            page_count  INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'pending', -- pending / done / failed
            error       TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def upsert_document(
    file_name: str,
    page_count: int = 0,
    chunk_count: int = 0,
    status: str = "done",
    error: str | None = None,
) -> None:
    """插入或更新一份文档记录。

    ★ 用 UPSERT 而不是 INSERT：同一份文件重复上传时不报错，直接更新。
      ON CONFLICT(file_name) DO UPDATE 是 SQLite 的语法。
    """
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO documents (file_name, page_count, chunk_count, status, error, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_name) DO UPDATE SET
            page_count  = excluded.page_count,
            chunk_count = excluded.chunk_count,
            status      = excluded.status,
            error       = excluded.error
        """,
        (file_name, page_count, chunk_count, status, error, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def list_documents() -> list[dict]:
    """列出全部文档，按上传时间倒序。前端文档列表页用。"""
    conn = get_conn()
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document(file_name: str) -> None:
    """删掉一条文档记录。注意：向量库里的 chunk 要另外删（见 vector_store.delete_by_source）。"""
    conn = get_conn()
    conn.execute("DELETE FROM documents WHERE file_name = ?", (file_name,))
    conn.commit()
    conn.close()
