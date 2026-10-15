import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agent import loop
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalysisRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="分析需求")
    max_steps: int | None = Field(
        default=None, ge=1, le=20, description="Agent 最大步数，不填用配置里的默认值"
    )


@router.post("/analysis", tags=["analysis"])
async def analyze(req: AnalysisRequest):
    """Agent 分析接口。

    实现：调 app.agent.loop 的手写循环。

    返回：
        {"answer": "...", "steps": [...], "stopped_reason": "completed"}

    ★ 把 steps 也返回给前端 —— 能展示"Agent 做了哪几步"，
      这是演示时的亮点，面试官能直观看到 Agent 在工作。
    """
    #入参校验兜底：万一前端传了个 1000，
    max_steps = req.max_steps or settings.max_agent_steps

    try:
        # ★ Agent 循环里全是同步的模型调用和工具执行，必须丢线程池。
        #   而且它耗时很长（好几轮 LLM 调用），前端请求超时要设得足够大。
        result = await run_in_threadpool(loop.run_agent, req.question, max_steps)
    except Exception as e:
        logger.exception("Agent 分析失败")
        raise HTTPException(status_code=502, detail=f"分析服务异常：{e}")

    return result


@router.post("/analysis/graph", tags=["analysis"])
async def analyze_graph(req: AnalysisRequest):
    # LangGraph 状态图版本。
    try:
        from app.agent.graph import build_graph
    except ImportError as e:
        raise HTTPException(status_code=501, detail=f"LangGraph 未安装：{e}")

    try:
        state = await run_in_threadpool(
            build_graph().invoke,
            {"question": req.question, "needs_retrieval": False, "hits": [], "answer": "", "extracted": {}},
        )
    except Exception as e:
        logger.exception("图编排分析失败")
        raise HTTPException(status_code=502, detail=f"图分析服务异常：{e}")

    # ★ 返回结构和 /analysis 保持一致（answer + stopped_reason），
    #   前端可以复用同一套渲染逻辑，不用为两个接口写两种展示代码。
    return {
        "answer": state.get("answer", ""),
        "needs_retrieval": state.get("needs_retrieval"),
        "hit_count": len(state.get("hits") or []),
        "extracted": state.get("extracted") or {},
        "stopped_reason": "completed",
    }
