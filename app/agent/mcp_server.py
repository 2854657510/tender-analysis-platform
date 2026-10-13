"""MCP Server —— D8 任务。

【要做什么】把平台的 3 个能力封装成 MCP 工具，能被任意 MCP 客户端调用。

【MCP 和 Function Calling 什么关系】（面试必问）
    Function Calling 是"模型能调函数"的【能力】
    MCP 是"函数怎么被标准化描述和发现"的【协议】
    → 有了 MCP，你写一次工具，任何支持 MCP 的客户端都能用，不用为每个应用重写

【依赖】pip install mcp

【怎么验证】
    MCP 官方提供了一个调试工具 Inspector：
        npx @modelcontextprotocol/inspector python app/agent/mcp_server.py
    在网页界面里能看到你注册的工具，并能直接调用 —— 截图存好，面试要用。
"""
from mcp.server.fastmcp import FastMCP

from app.agent.tools import compare_budgets, extract_fields, search_documents

mcp = FastMCP("tender-analysis")


# ---------------------------------------------------------------------------
# 注册工具。FastMCP 用装饰器把普通函数变成 MCP 工具 ——
# 函数的 docstring 会变成工具的 description，【写给模型看的，要写清楚】。
# ---------------------------------------------------------------------------

@mcp.tool()
def tool_search_documents(query: str, top_k: int = 5) -> dict:
    """在招投标文档库中检索相关内容。需要查找项目预算、截止时间、资格要求等信息时使用。

    Args:
        query: 检索关键词或问题
        top_k: 返回条数，默认 5
    """
    return search_documents(query, top_k)


@mcp.tool()
def tool_extract_fields(document: str) -> dict:
    """从公告原文中抽取结构化字段（项目名称、采购人、预算等）。拿到完整原文需要结构化时使用。

    Args:
        document: 公告原文
    """
    return extract_fields(document)


@mcp.tool()
def tool_compare_budgets(projects: list[str]) -> dict:
    """对比多个项目的预算金额并排序。用户想知道"哪个项目预算最高"时使用。

    Args:
        projects: 要对比的项目名称列表
    """
    return compare_budgets(projects)


if __name__ == "__main__":
    # stdio 模式：通过标准输入输出通信，适合本地客户端接入
    mcp.run()

# D8 验收：
#   · 用 Inspector 能看到并调用这 3 个工具
#   · tool_extract_fields 能真的调到项目1的抽取能力（两个项目连通了）
#   · 能说清 MCP 和 Function Calling 的分工
