"""手写 Agent 循环 —— D6 任务 ★★★ 面试高频

【为什么不用框架手写一遍】面试官最爱问"Function Calling 的原理是什么"。
    能用框架搭出来的人很多，能白板写出循环的人很少。
    「我手写过 Agent 循环」比「我会用 LangChain」更有分量。

【ReAct 范式：Thought → Action → Observation】
    模型想 → 模型说要调什么工具 → 你执行 → 把结果喂回去 → 模型再想 → ...
    直到模型说"我有答案了"，或者达到最大步数。
"""
import json
import logging

from app.agent.tools import TOOLS, TOOL_FUNCTIONS
from app.config import settings
from modules.llm.client import chat

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


def run_agent(question: str, max_steps: int | None = None) -> dict:
    """Agent 主循环。

    返回：
        {
            "answer": "最终答案",
            "steps": [{"step": 1, "tool": "search_documents", "args": {...}, "result": {...}}, ...],
            "stopped_reason": "completed" | "max_steps",
        }

    ★★★ 实现步骤（照着写）：

        1. 初始化 messages：
             messages = [
                 {"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": question},
             ]
           steps = []

        2. 循环 for step in range(1, max_steps + 1)：

           a) 调模型，【必须带上 tools 参数】：
                resp = client.chat.completions.create(
                    model=..., messages=messages,
                    tools=TOOLS,                    # ← 关键，不传模型就不知道有工具
                    tool_choice="auto",             # ← 让模型自己决定用不用
                )
              ★ 注意：这一步不能走 modules/llm/client.py 的 chat()，
                因为那个封装没传 tools。要么扩展 chat()，要么这里直接调 SDK。

           b) 取出 message = resp.choices[0].message，把【整个 message】append 进 messages
                messages.append(message)
              ★ 为什么是整个 message 而不是自己拼一个 dict：
                正常的 tool_calls 必须原样传回去，否则下一轮模型会报错。

           c) 判断 message.tool_calls 是否为空：
                - 为空 → 模型给了最终答案，return {"answer": message.content, "stopped_reason": "completed"}
                - 不为空 → 继续 d)

           d) 对每个 tool_call：
                - fn_name = tool_call.function.name
                - args = json.loads(tool_call.function.arguments)   # ← 参数是 JSON 字符串
                - result = TOOL_FUNCTIONS[fn_name](**args)
                - messages.append({
                      "role": "tool",
                      "tool_call_id": tool_call.id,       # ← 必须对应上，否则模型不知道这是谁的返回值
                      "content": json.dumps(result, ensure_ascii=False),
                  })
                - steps.append({...})  # 记录这一步，方便调试和展示

        3. 循环走完还没返回 → return {"answer": "达到最大步数仍未得出结论", "stopped_reason": "max_steps"}

    ★★★ 为什么必须有 max_steps：
        模型可能反复调用同一个工具、或陷入循环。
        没有上界的话，一次异常就能把你的 API 额度烧光。
        这和项目1里的 max_rounds 是同一个道理 ——
        【Agent 开发里所有循环都必须有终止条件】。

    D6 验收：
        · 问一个需要检索的问题，能看到 Agent 真的调用了 search_documents
        · 打印 steps，能看到完整的"想 → 调 → 看结果 → 再想"过程
        · ★ 构造一个会让模型反复调用工具的问题，验证 max_steps 生效
        · ★★ 能白板手写出这个循环的骨架
    """
    raise NotImplementedError("D6 任务：实现 Agent 循环")


def run_agent_stream(question: str):
    """带过程输出的 Agent 循环（可选）。

    每完成一步就 yield 一次，前端可以实时显示"正在检索..."
    面试演示效果好很多。
    """
    raise NotImplementedError("D6 任务（可选）：流式 Agent 循环")
