import logging

from modules.extraction import extract as extraction
from modules.retrieval import hybrid

logger = logging.getLogger(__name__)
# 格式必须严格符合 Function Calling 规范，字段一个都不能少：
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "在已上传的招投标文档库中检索相关内容。"
                "当你需要查找某个项目的预算、截止时间、资格要求等具体信息时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_fields",
            "description": (
                "从一段公告原文中抽取结构化字段（项目名称、采购人、预算等）。"
                "当你拿到一段完整原文、需要结构化信息时使用，比 search_documents 更精确。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document": {"type": "string", "description": "公告原文"},
                },
                "required": ["document"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_budgets",
            "description": "对比多个项目的预算金额，返回排序结果。当用户想知道'哪个项目预算最高'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "projects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的项目名称列表",
                    },
                },
                "required": ["projects"],
            },
        },
    },
]


def search_documents(query: str, top_k: int = 5) -> dict:
    """执行检索工具。"""
    hits = hybrid.search(query, top_k=top_k)
    return {"results": hits, "count": len(hits)}


def extract_fields(document: str) -> dict:
    # 执行抽取工具
    return extraction.extract_safe(document)


def compare_budgets(projects: list[str]) -> dict:
    # 执行预算对比工具。
    rows = []

    for name in projects:
        row = {"project": name, "budget_wan": None, "source": None, "page": None}

        try:
            # ① 先检索这个项目的相关原文
            hits = hybrid.search(name, top_k=3)

            if hits:
                row["source"] = hits[0].get("source")
                row["page"] = hits[0].get("page")

                # ② 把前几条原文拼起来交给抽取器。
                merged = "\n".join(h["text"] for h in hits)

                # ③ 复用项目1 的抽取能力，并走它的降级路径
                extracted = extraction.extract_safe(merged)
                row["budget_wan"] = extracted.get("budget_wan")

        except Exception as e:
            # 单个项目失败不中断整体
            logger.warning("项目「%s」预算提取失败：%s", name, e)

        rows.append(row)

    #排序：有钱的排前面、按金额从高到低；没查到预算的沉到最后。
    rows.sort(key=lambda r: (r["budget_wan"] is None, -(r["budget_wan"] or 0)))

    return {
        "comparison": rows,
        "count": len(rows),
        "note": "budget_wan 为 null 表示在所上传文档中未找到该项目预算",
    }


# 工具名 → 执行函数的映射。Agent 循环靠它把模型的"我要调用 X"变成真的函数调用。
TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "extract_fields": extract_fields,
    "compare_budgets": compare_budgets,
}
