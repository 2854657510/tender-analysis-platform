"""Agent 工具定义 —— D6 任务。

【要做什么】定义 Agent 能调用的工具，并实现每个工具的执行逻辑。

【工具设计的关键】工具的 description 是【写给模型看的】。
模型靠它判断"什么时候该用这个工具" —— description 写得含糊，模型就选错工具。
这和项目1里 Field 的 description 是同一个道理。
"""
from modules.extraction import extract as extraction
from modules.retrieval import hybrid


# ---------------------------------------------------------------------------
# 工具定义（发给模型的 JSON Schema）
#
# ★ 格式必须严格符合 Function Calling 规范，字段一个都不能少：
#     type / function.name / function.description / function.parameters
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 工具的执行实现
# ---------------------------------------------------------------------------

def search_documents(query: str, top_k: int = 5) -> dict:
    """执行检索工具。"""
    hits = hybrid.search(query, top_k=top_k)
    return {"results": hits, "count": len(hits)}


def extract_fields(document: str) -> dict:
    """执行抽取工具 —— 直接复用项目1的能力。

    ★ 这是两个项目真正连起来的地方：D8 之后它还会被封装成 MCP 工具。
    """
    return extraction.extract_safe(document)


def compare_budgets(projects: list[str]) -> dict:
    """执行预算对比工具。

    实现提示：
        1. 对每个项目名调 search_documents，找到它的预算
        2. 排序返回
    """
    raise NotImplementedError("D6 任务：实现预算对比工具")


# 工具名 → 执行函数的映射。Agent 循环靠它把模型的"我要调用 X"变成真的函数调用。
TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "extract_fields": extract_fields,
    "compare_budgets": compare_budgets,
}
