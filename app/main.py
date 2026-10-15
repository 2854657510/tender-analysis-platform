
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

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


# ---------------------------------------------------------------------------
# D11：限流中间件（滑动窗口）
#
# 【为什么必须有限流】这个平台每个请求背后都是一次或多次【要花钱的模型调用】。
#   不加限制的话，一个循环脚本、或者刷新键按多了，就能把额度打光。
#   "接口不设防"是个人项目部署到公网后最常见的翻车方式。
#
# 【为什么放在中间件而不是各个接口里】
#   限流是 HTTP 层的事，和业务逻辑没关系。
#   放中间件里，以后新增接口自动被保护，不用每个函数都记得加一行。
# ---------------------------------------------------------------------------
RATE_LIMIT_MAX = 30       # 每个 IP 在窗口内最多多少次请求
RATE_LIMIT_WINDOW = 60    # 窗口长度（秒）
RATE_LIMIT_EXEMPT = {"/health", "/docs", "/openapi.json", "/redoc"}   # 这些不限流

# 每个 IP 一个 deque，存最近若干次请求的时间戳。
# ★ 用 deque 而不是 list：滑动窗口要频繁地从左边丢过期的记录，
#   deque.popleft() 是 O(1)，list 要 O(n)。
_rate_hits: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """按客户端 IP 做滑动窗口限流。

    ★ 和缓存一样，这是【进程内】限流：uvicorn 起多个 worker 时，
      每个 worker 各算各的，实际放行量 = 上限 × worker 数。
      生产环境该用 Redis 做集中计数。这里先用内存版演示思路。
    """

    async def dispatch(self, request: Request, call_next):
        # 健康检查和文档页不参与限流 ——
        # 否则你拿 /health 做监控探测，反而会把自己限掉
        if request.url.path in RATE_LIMIT_EXEMPT:
            return await call_next(request)

        # ★ 取不到 IP 时用 "unknown" 兜底，而不是跳过限流。
        #   宁可把来源不明的一起限住，也不要留一个完全不设防的口子。
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        with _rate_lock:
            hits = _rate_hits[client_ip]
            # ① 把窗口外的旧记录丢掉（这才是"滑动"的含义）
            while hits and now - hits[0] > RATE_LIMIT_WINDOW:
                hits.popleft()
            # ② 窗口内已经打满了 → 拒绝
            if len(hits) >= RATE_LIMIT_MAX:
                retry_after = int(RATE_LIMIT_WINDOW - (now - hits[0])) + 1
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"请求太频繁，请 {retry_after} 秒后重试"},
                    # ★ Retry-After 是 HTTP 标准头，正经的客户端会照着它退避重试
                    headers={"Retry-After": str(retry_after)},
                )
            # ③ 记账
            hits.append(now)

        return await call_next(request)


app = FastAPI(
    title="智能标书分析平台",
    version="0.1.0",
    description="招投标文档解析 · RAG 检索问答 · Agent 分析",
    lifespan=lifespan,
)

# ★ 中间件的顺序有讲究：先加的后被"包在外层"。
#   限流放最外层，这样被限掉的请求根本不会走到 CORS 和其它逻辑，最省资源。
app.add_middleware(RateLimitMiddleware)

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
