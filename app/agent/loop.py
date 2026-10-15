
import json
import logging

from app.agent.tools import TOOLS, TOOL_FUNCTIONS
from app.config import settings
from modules.llm.client import chat_with_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个招投标文档分析助手，可以调用工具来查找和分析信息。

工作方式：
1. 先判断需不需要调用工具。如果需要，调用合适的工具获取信息。
2. 拿到工具结果后，判断信息是否足够回答问题。
3. 信息足够了，就用中文给出简洁的答案。

规则：
- 只依据工具返回的内容回答，不要编造
- 工具返回空结果时，如实说明"没有找到相关信息"
"""

# steps 里每条记录的 result 最多留多长（字符）。
STEP_RESULT_CHARS = 600


def _message_to_dict(message) -> dict:
    """把模型返回的 message 对象转成【可以再发回去】的 dict。

      模型返回的 message 里，除了 content 还有 tool_calls。
      下一轮请求必须把这个 tool_calls 【原样带回去】，否则模型会报错
      （它会说"前面那条 assistant 消息引用了不存在的工具调用"）。

      所以不能只 append {"role": "assistant", "content": message.content} ——
      那样 tool_calls 就丢了，循环第二轮必崩。

    ★ assistant 消息里 content 可能为 None，
      这里保留 None 而不是转成空串，是为了忠于原始响应。
    """
    data: dict = {"role": "assistant", "content": message.content}

    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.id,                # ← 这个 id 下一轮要和 tool 消息里的 tool_call_id 对上
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    # ★ arguments 是【JSON 字符串】，不是 dict。
                    #   这里不能 json.loads —— 要原样传回给模型。
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]

    return data


def _execute_tool(tool_call) -> tuple[dict, dict]:
    """执行一个工具调用，返回 (给模型看的 result, 记进 steps 的摘要)。

    工具出错【不 raise】，而是把错误信息当成"观察结果"喂回模型。

    """
    fn_name = tool_call.function.name
    raw_args = tool_call.function.arguments

    # ① 解析参数。模型给的是 JSON 字符串，而且偶尔会给残缺的 JSON
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as e:
        logger.warning("工具参数不是合法 JSON：%s", raw_args)
        return {"error": f"参数不是合法的 JSON：{e}。请重新给出正确的参数。"}, {
            "tool": fn_name, "args": raw_args, "error": "JSON 解析失败",
        }

    # ② 找工具。模型偶尔会幻觉出一个不存在的工具名
    func = TOOL_FUNCTIONS.get(fn_name)
    if func is None:
        logger.warning("模型请求了不存在的工具：%s", fn_name)
        return {
            "error": f"不存在名为 {fn_name} 的工具。可用工具：{list(TOOL_FUNCTIONS)}",
        }, {"tool": fn_name, "args": args, "error": "工具不存在"}

    # 用 TypeError 单独接住"参数不对"这类模型的问题，
    # 和"工具内部报错"（我的问题）区分开 —— 日志里一眼能看出该改哪边
    try:
        result = func(**args)
        return result, {"tool": fn_name, "args": args, "result": result}

    except TypeError as e:
        logger.warning("工具 %s 参数不匹配：%s", fn_name, e)
        return {"error": f"调用 {fn_name} 的参数不正确：{e}"}, {
            "tool": fn_name, "args": args, "error": f"参数错误：{e}",
        }

    except Exception as e:
        logger.exception("工具 %s 执行失败", fn_name)
        return {"error": f"{fn_name} 执行失败：{e}"}, {
            "tool": fn_name, "args": args, "error": str(e)[:200],
        }


def run_agent(question: str, max_steps: int | None = None) -> dict:
    """Agent 主循环。

    返回：
        {
            "answer": "最终答案",
            "steps": [{"step": 1, "tool": "search_documents", "args": {...}, "result": {...}}, ...],
            "stopped_reason": "completed" | "max_steps",
        }
    """
    max_steps = max_steps or settings.max_agent_steps

    # ① 初始化对话。system 定人设和规则，user 提问题
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list[dict] = []

    # ② ★ 主循环。for + 计数器
    for step in range(1, max_steps + 1):
        logger.info("Agent 第 %s/%s 步", step, max_steps)

        message = chat_with_tools(messages, TOOLS, tool_choice="auto")

        # 整个 message 收进历史（含 tool_calls），下一轮要原样带回
        messages.append(_message_to_dict(message))

        if not message.tool_calls:
            logger.info("Agent 在第 %s 步给出最终答案", step)
            return {
                "answer": message.content or "",
                "steps": steps,
                "stopped_reason": "completed",
            }

        # ④ 有 tool_calls：逐个执行，把结果喂回去
        for tool_call in message.tool_calls:
            result, summary = _execute_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    # ★★ tool_call_id 必须和上面 assistant 消息里的 id 严格对应。
                    "tool_call_id": tool_call.id,
                    # ★ 内容必须是【字符串】。dict 要先 dump 成 JSON。
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

            step_record = {
                "step": step,
                "tool": tool_call.function.name,
                "args": summary.get("args"),
            }
            if "result" in summary:
                step_record["result"] = json.dumps(
                    summary["result"], ensure_ascii=False, default=str
                )[:STEP_RESULT_CHARS]
            else:
                step_record["error"] = summary.get("error")

            steps.append(step_record)
            logger.info("  第 %s 步调用工具：%s", step, tool_call.function.name)
    logger.warning("Agent 达到最大步数 %s，强制结束", max_steps)
    return {
        "answer": f"已达到最大步数（{max_steps} 步）仍未得出结论，请把问题拆得更具体一些。",
        "steps": steps,
        "stopped_reason": "max_steps",
    }


def run_agent_stream(question: str, max_steps: int | None = None):
    # 带过程输出的 Agent 循环

    max_steps = max_steps or settings.max_agent_steps

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    yield {"type": "start", "question": question}

    for step in range(1, max_steps + 1):
        message = chat_with_tools(messages, TOOLS, tool_choice="auto")
        messages.append(_message_to_dict(message))

        if not message.tool_calls:
            yield {"type": "answer", "step": step, "answer": message.content or ""}
            return

        for tool_call in message.tool_calls:
            # ★ 先推"开始调用"，再推"调用完成"
            yield {
                "type": "action",
                "step": step,
                "tool": tool_call.function.name,
                "args": tool_call.function.arguments,
            }

            result, _ = _execute_tool(tool_call)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

            yield {
                "type": "observation",
                "step": step,
                "tool": tool_call.function.name,
                "result": json.dumps(result, ensure_ascii=False, default=str)[:STEP_RESULT_CHARS],
            }

    yield {"type": "answer", "answer": f"已达到最大步数（{max_steps} 步）仍未得出结论。",
           "stopped_reason": "max_steps"}
