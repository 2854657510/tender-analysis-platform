"""分析接口 —— D12 任务。

【这个接口是 Agent 的出口】
    复杂问题（"对比这几个项目的资格要求差异"）走 Agent 编排，
    由 Supervisor 拆解成多步，调用检索/抽取工具完成。
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class AnalysisRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="分析需求")


@router.post("/analysis", tags=["analysis"])
async def analyze(req: AnalysisRequest):
    """Agent 分析接口。

    实现：调 app.agent.graph 的图（D7 之后），或 app.agent.loop 的循环（D6）

    返回：
        {"answer": "...", "steps": [...], "stopped_reason": "completed"}

    ★ 把 steps 也返回给前端 —— 能展示"Agent 做了哪几步"，
      这是演示时的亮点，面试官能直观看到 Agent 在工作。
    """
    raise NotImplementedError("D12 任务：实现分析接口")
