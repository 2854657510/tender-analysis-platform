import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from modules.llm import client as llm_client
from modules.retrieval import rag

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索条数")


@router.post("/query", tags=["query"])
async def query(req: QueryRequest):
    """RAG 问答主接口。

    返回：{"answer": "...", "sources": [...]}

    ★ 用 run_in_threadpool 包一层 —— rag.answer 是同步阻塞调用，
      在 async def 里直接调会卡死整个事件循环（项目1踩过这个坑）。

    D12 验收：前端能提问并看到带出处的回答
    """
    try:
        # ★ 这里不只是一次模型调用：内部还包含 Embedding 推理、Chroma 查询、
        #   BM25 建索引、rerank 打分。全是同步阻塞操作，必须丢线程池。
        return await run_in_threadpool(rag.answer, req.question, top_k=req.top_k)
    except Exception as e:
        logger.exception("问答失败")
        # ★ 对外返回 502 而不是把原始异常抛给前端：
        #   异常的堆栈里可能带路径、内部结构，不该暴露给调用方。
        #   详细信息记在日志里，用户拿到一句人能看懂的话。
        raise HTTPException(status_code=502, detail=f"问答服务异常：{e}")


@router.post("/query/stream", tags=["query"])
async def query_stream(req: QueryRequest):
    """流式问答（可选实现）。

    ★ 和 /query 的区别：模型边生成边推给前端，用户不用干等十几秒
      看着一个转圈 —— 首字延迟从"整段生成完"降到"第一个 token"。
      这是体感上最明显的优化，面试演示时非常加分。
    """
    # ① 先做检索（同步阻塞 → 线程池）。检索完才轮到模型生成。
    try:
        hits = await run_in_threadpool(
            rag._retrieve, req.question, False, req.top_k
        )
    except Exception as e:
        logger.exception("流式问答的检索阶段失败")
        raise HTTPException(status_code=502, detail=f"检索失败：{e}")

    if not hits:
        # 没检索到就别调模型了（同 rag.answer 的处理）
        async def _empty():
            yield f"data: {json.dumps({'type': 'answer', 'answer': '知识库中没有检索到相关内容。'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(_empty(), media_type="text/event-stream")

    context, sources = rag.build_context(hits)
    prompt = rag.render(rag.ANSWER_PROMPT, context=context, question=req.question)

    async def event_generator():
        # ② ★ 先把出处推给前端。为什么放最前面：
        #   前端可以立刻把"参考资料"列表渲染出来，用户马上看到"它找到了这些"，
        #   然后再看着答案一个字一个字长出来 —— 等待感大幅降低。
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

        try:
            # ③ 异步流式调用。注意这里用的是 aclient（异步客户端），
            #   不能用同步的 —— 用错会把事件循环卡死。
            stream = await llm_client.stream([{"role": "user", "content": prompt}])
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type': 'delta', 'text': delta}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("流式生成中断")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"

        # ④ 结束标记。★ 前端必须靠它知道"流完了"，
        #   否则会一直保持着加载状态。
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    # ★ media_type 必须是 text/event-stream，浏览器才会按 SSE 处理。
    #   X-Accel-Buffering: no 是给 Nginx 看的 —— 不加的话 Nginx 会把
    #   整个响应缓冲起来再一次性发，流式就变成"假流式"了（D13 部署时的坑）。
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
