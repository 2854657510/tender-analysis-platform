"""问答接口 —— D12 任务（D3 就会开始用到）。

【两个接口】
    POST /api/query          普通问答，返回完整答案
    POST /api/query/stream   流式问答（SSE），边生成边返回

【流式怎么实现】参考项目1的 app/api/stream.py，改成调用 RAG 链路即可。
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条数")


@router.post("/query", tags=["query"])
async def query(req: QueryRequest):
    """RAG 问答主接口。

    实现：直接调 modules.retrieval.rag.answer(req.question)
    返回：{"answer": "...", "sources": [...]}

    ★ 用 run_in_threadpool 包一层 —— rag.answer 是同步阻塞调用，
      在 async def 里直接调会卡死整个事件循环（项目1踩过这个坑）。

    D12 验收：前端能提问并看到带出处的回答
    """
    raise NotImplementedError("D12 任务：实现问答接口")
