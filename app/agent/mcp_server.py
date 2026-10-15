from mcp.server.fastmcp import FastMCP

from app.agent.tools import compare_budgets, extract_fields, search_documents

mcp = FastMCP("tender-analysis")

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
    """对比多个项目的预算金额并排序。

    Args:
        projects: 要对比的项目名称列表
    """
    return compare_budgets(projects)


if __name__ == "__main__":
    # stdio 模式：通过标准输入输出通信，适合本地客户端接入
    mcp.run()
