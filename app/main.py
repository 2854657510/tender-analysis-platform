"""应用入口。

【现状】这个文件【可以直接运行】—— 启动后能访问 /health 和 /docs。
       但三个业务接口都还是 NotImplementedError，那是你 D1–D12 要填的。
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.api import analysis_api, documents_api, query_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建表。

    ★ 用 lifespan 而不是 @app.on_event("startup") —— 后者在新版 FastAPI 里已废弃。
    """
    db.init_db()
    logging.getLogger(__name__).info("数据库已就绪")
    yield


app = FastAPI(
    title="智能标书分析平台",
    version="0.1.0",
    description="招投标文档解析 · RAG 检索问答 · Agent 分析",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # ★ 前端 Streamlit 调后端需要，不加会被浏览器拦掉
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_api.router, prefix="/api", tags=["documents"])
app.include_router(query_api.router, prefix="/api", tags=["query"])
app.include_router(analysis_api.router, prefix="/api", tags=["analysis"])


@app.get("/health", tags=["ops"])
async def health():
    """健康检查。部署后用这个确认服务活着。"""
    return {"status": True}
