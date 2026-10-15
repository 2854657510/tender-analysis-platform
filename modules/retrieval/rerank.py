
import json
import logging

from app.config import settings
from modules.llm.client import chat

logger = logging.getLogger(__name__)

# 每条候选喂给模型时最多截取多少字。
MAX_DOC_CHARS = 300

RERANK_PROMPT = """你是一个检索结果相关性评分助手。
用户问题：{query}

下面是若干候选文档片段，请给每一个片段打一个 0-10 的相关度分数：
- 10 分：直接回答了问题
- 5 分：部分相关，能提供背景信息
- 0 分：完全无关

只输出 JSON，格式为 {{"scores": [{{"id": 1, "score": 8}}, ...]}}，不要任何解释。

候选片段：
{documents}"""


def _score_with_llm(query: str, hits: list[dict]) -> dict[int, float]:
    """调一次模型，拿到 {文档序号: 分数}。
    """
    lines = []
    for i, hit in enumerate(hits, start=1):
        snippet = " ".join(hit["text"][:MAX_DOC_CHARS].split())
        lines.append(f"[{i}] {snippet}")

    prompt = RERANK_PROMPT.format(query=query, documents="\n".join(lines))
    # json_mode=True → 强制模型输出合法 JSON
    raw = chat([{"role": "user", "content": prompt}], json_mode=True, max_tokens=1024)
    data = json.loads(raw)

    #  兼容两种常见的返回形态：
    #   ① {"scores": [{"id": 1, "score": 8}, ...]}
    #   ② [{"id": 1, "score": 8}, ...]
    rows = data.get("scores", []) if isinstance(data, dict) else data

    scores = {}
    for row in rows:
        try:
            scores[int(row["id"])] = float(row["score"])
        except (KeyError, TypeError, ValueError):
            # 单条解析失败就跳过，不影响其它条目
            continue
    return scores


def rerank(query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
    """对召回结果重新排序。

    输入：hybrid.search() 的输出（20 条左右）
    输出：精排后的 top_k 条，每项加一个 "rerank_score" 字段

    降级兜底：Rerank 服务挂了【不能整个崩】，要能退回 RRF 的顺序。
    写个 try/except，失败时直接返回原顺序
    """
    top_k = top_k or settings.rerank_top_k

    if not hits:
        return []

    try:
        scores = _score_with_llm(query, hits)

        for i, hit in enumerate(hits, start=1):
            # 模型漏掉的条目用一个很低的分数兜住，
            hit["rerank_score"] = scores.get(i, -1.0)

        ordered = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        logger.info("Rerank 完成：%s 条 → 取前 %s 条", len(ordered), top_k)
        return ordered[:top_k]

    except Exception as e:
        # 降级路径：任何异常（网络、限流、模型输出不是 JSON）都在这里被接住。
        logger.warning("Rerank 失败，降级为原顺序：%s: %s", type(e).__name__, e)
        return hits[:top_k]
