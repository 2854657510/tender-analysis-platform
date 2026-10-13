"""模型调用层 —— 所有跟大模型说话的地方都走这里。

【为什么要单独一层】
1. 换模型只改这一个文件（比如从 DeepSeek 换成通义，只改 base_url 和 model）
2. 重试/日志/计费统计集中在一处，不用在每个业务函数里重复写
3. 业务代码不直接 import openai，测试时好替换成假客户端

【来源】重试逻辑从 ai-agent-portfolio 的 app/extract.py 搬过来，做成了通用版。
"""
import logging
import random
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 两个客户端：同步 + 异步
#   同步：脚本、批处理、Agent 循环（D6）
#   异步：FastAPI 的流式接口（D2 那个 SSE）
# 用错了会卡死事件循环，别混。
# ---------------------------------------------------------------------------
client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
aclient = AsyncOpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

# 只有这几类错误值得重试。
# 判据：4xx 是我的问题（重试一万次还是失败）；5xx 和 429 是对方的问题（等会儿再来）。
RETRYABLE_STATUS = {429, 500, 503}


def call_with_retry(fn, *args, max_retries: int = 5, base_delay: float = 1.0, **kwargs):
    """带【指数退避 + 抖动】的重试。

    指数退避：第 1 次失败等 1s，第 2 次 2s，第 3 次 4s。对方已趴下，猛冲只会更糟。
    抖动 jitter：加一点随机。没有它，一堆客户端会在同一毫秒一起重试，把服务再打垮。
    """
    err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:            # 429
            err = e
        except APIStatusError as e:            # 其他 HTTP 错误
            if e.status_code not in RETRYABLE_STATUS:
                raise                          # 400/401/402/422 直接抛，不浪费时间
            err = e
        except APIConnectionError as e:        # 网络层失败
            err = e

        if attempt == max_retries - 1:
            raise err

        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        logger.warning("第 %s 次失败(%s)，%.1fs 后重试", attempt + 1, err, delay)
        time.sleep(delay)


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    json_mode: bool = False,
    disable_thinking: bool = True,
) -> str:
    """同步调一次模型，返回文本。最常用的入口。

    参数说明：
        json_mode        True 时加 response_format，保证返回合法 JSON
        disable_thinking True 时关掉思考模式。
                         【为什么默认关】实测关掉后输出 token 从 1016 降到 78（降 93%），
                         而且 temperature 在思考模式下【不生效】。
                         RAG 回答和字段抽取都不需要长推理，开着纯属烧钱。
    """
    kwargs = {
        "model": settings.deepseek_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if disable_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    resp = call_with_retry(client.chat.completions.create, **kwargs)

    # 打 token 用量，慢慢就对"一次调用大概多少钱"有感觉了
    if resp.usage:
        logger.info(
            "token：prompt=%s completion=%s 缓存命中=%s",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            getattr(resp.usage, "prompt_cache_hit_tokens", "n/a"),
        )

    return resp.choices[0].message.content or ""


def stream(messages: list[dict], *, max_tokens: int = 2048):
    """异步流式调用。用法见 app/api/stream.py（从项目1复制过来改成这里即可）。"""
    return aclient.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
    )
