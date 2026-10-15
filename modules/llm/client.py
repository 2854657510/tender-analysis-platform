
import hashlib
import json
import logging
import random
import threading
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
    use_cache: bool = True,
) -> str:
    """同步调一次模型，返回文本。最常用的入口。

    参数说明：
        json_mode        True 时加 response_format，保证返回合法 JSON
        disable_thinking True 时关掉思考模式。
                         【为什么默认关】实测关掉后输出 token 从 1016 降到 78（降 93%），
                         而且 temperature 在思考模式下【不生效】。
                         RAG 回答和字段抽取都不需要长推理，开着纯属烧钱。
        use_cache        D11 新增：命中缓存就直接返回，不再花钱调模型。
                         只有 temperature=0（确定性输出）时才真正启用 —— 见下方说明。
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

    # ---------------- D11：结果缓存 ----------------
    # ★ 为什么缓存放在这里：chat() 是【全项目所有模型调用的唯一收口】。
    #   在这一层缓存，等于 RAG 问答、字段抽取、查询改写全部自动受益，
    #   业务代码一行都不用改 —— 这就是"选对收口层"的价值。
    #
    # ★ 只在 temperature=0 时缓存：temperature>0 本来就是要每次不一样的输出，
    #   缓存下来等于把随机性抹掉，反而破坏语义。
    cache_key = None
    if use_cache and temperature == 0:
        cache_key = _make_cache_key(kwargs)
        hit = _cache_get(cache_key)
        if hit is not None:
            logger.info("【命中缓存】省下一次模型调用")
            return hit

    # ---------------- D11：耗时日志 ----------------
    started = time.perf_counter()
    resp = call_with_retry(client.chat.completions.create, **kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # 打 token 用量，慢慢就对"一次调用大概多少钱"有感觉了
    if resp.usage:
        logger.info(
            "耗时=%.0fms token：prompt=%s completion=%s 缓存命中=%s",
            elapsed_ms,
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
            getattr(resp.usage, "prompt_cache_hit_tokens", "n/a"),
        )

    content = resp.choices[0].message.content or ""

    if cache_key is not None:
        _cache_put(cache_key, content)

    return content


# ---------------------------------------------------------------------------
# 模型回复缓存（D11）
#
# 【局限，面试要主动说出来】
#   这是【进程内】缓存：uvicorn 起多个 worker（--workers 2）时，
#   每个 worker 各有一份，缓存不共享、命中率减半；
#   进程重启缓存就没了。
#   生产环境该换成 Redis —— 那是跨进程、可持久化、还能设 TTL 的方案。
#   这里用内存版是为了零依赖演示这个思路。
# ---------------------------------------------------------------------------
_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
CACHE_MAX_SIZE = 256      # ★ 必须有上限，否则长跑的服务会一直吃内存


def _make_cache_key(kwargs: dict) -> str:
    """把请求参数压成一个哈希，作为缓存键。

    ★ 要覆盖【所有影响输出的参数】（messages / temperature / max_tokens ...），
      漏掉任何一个都会出现"改了参数却拿到旧答案"的诡异 bug。
      sort_keys=True 保证字典顺序变化不会算出不同的键。
    """
    payload = json.dumps(kwargs, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    with _CACHE_LOCK:
        return _CACHE.get(key)


def _cache_put(key: str, value: str) -> None:
    with _CACHE_LOCK:
        # ★ 满了就整体清空。看着粗暴，但比实现 LRU 简单得多，
        #   而缓存本来就是"丢了也只是重算一遍"的东西，不需要精细淘汰。
        if len(_CACHE) >= CACHE_MAX_SIZE:
            _CACHE.clear()
            logger.info("缓存已满，整体清空")
        _CACHE[key] = value


def cache_stats() -> dict:
    """看一眼缓存现状（调试和演示用）。"""
    with _CACHE_LOCK:
        return {"size": len(_CACHE), "max_size": CACHE_MAX_SIZE}


def chat_with_tools(
    messages: list,
    tools: list[dict],
    *,
    tool_choice: str = "auto",
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """带工具调用的模型请求（D6 新增）。

    ★ 和 chat() 的区别：这里返回【整个 message 对象】，不是只返回文本字符串。

    为什么必须这样：Function Calling 的返回不止有 content，还有 tool_calls
    （模型说"我要调用哪个函数、参数是什么"）。
    只返回 content 的话，tool_calls 就丢了，Agent 循环根本没法往下走。

    ★ 为什么不在这里做缓存：Agent 的每一轮都带着上一轮的工具结果，
      messages 几乎从不重复，缓存命中率接近 0，白白增加内存占用。

    参数：
        tool_choice="auto"  让模型自己决定"用不用工具、用哪个"。
                            可选值还有 "none"（禁用）或
                            {"type":"function","function":{"name":"xxx"}}（强制指定）
    """
    resp = call_with_retry(
        client.chat.completions.create,
        model=settings.deepseek_model,
        messages=messages,
        tools=tools,              # ★ 不传这个，模型根本不知道有工具可用
        tool_choice=tool_choice,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )

    if resp.usage:
        logger.info(
            "Agent 一轮 token：prompt=%s completion=%s",
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens,
        )

    return resp.choices[0].message


def stream(messages: list[dict], *, max_tokens: int = 2048):
    """异步流式调用。用法见 app/api/stream.py（从项目1复制过来改成这里即可）。"""
    return aclient.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        stream=True,
        max_tokens=max_tokens,
    )
