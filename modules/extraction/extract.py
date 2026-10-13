import json
import logging
import random
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from app.config import settings
from modules.extraction.prompt import render
from modules.extraction.schema import TenderNotice

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)

EXAMPLES = [
    {
        # 干净的数据
        "input": "项目名称：XX市政务云平台建设。采购人：XX市大数据局。预算：约1200万元。",
        "output": (
            '{"project_name": "XX市政务云平台建设", "buyer": "XX市大数据局", '
            '"budget_wan": 1200.0, "deadline": null, "qualification": [], '
            '"contact": {"name": null, "phone": null}, "confidence": 0.9}'
        ),
    },
    {
        # 脏数据
        "input": "招标公告。本项目由某某单位组织，具体事宜详见附件。联系人：张工 电话138xxxx",
        "output": (
            '{"project_name": null, "buyer": null, "budget_wan": null, '
            '"deadline": null, "qualification": [], '
            '"contact": {"name": "张工", "phone": "138xxxx"}, "confidence": 0.4}'
        ),
    },
]

RETRYABLE_STATUS = {429, 500, 503}

# 带指数退避 + 抖动
def call_with_retry(fn, *args, max_retries: int = 5, base_delay: float = 1.0, **kwargs):
    err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:            # 429 请求太频繁
            err = e
        except APIStatusError as e:            # 其他 HTTP 状态错误
            if e.status_code not in RETRYABLE_STATUS:
                raise
            err = e
        except APIConnectionError as e:        # 网络层失败，值得重试
            err = e

        if attempt == max_retries - 1:
            raise err                        # 最后一次也失败了

        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
        logger.warning("第 %s 次失败(%s)，%.1fs 后重试", attempt + 1, err, delay)
        time.sleep(delay)


def _build_messages(document: str) -> list[dict]:
    schema_json = json.dumps(
        TenderNotice.model_json_schema(), ensure_ascii=False, indent=2
    )
    return [
        {
            "role": "system",
            "content": render(
                "extract",
                schema_json=schema_json,
                examples=EXAMPLES,
                document=document,
            ),
        }
    ]


def extract_with_retry(document: str, max_rounds: int = 3) -> TenderNotice:
    messages = _build_messages(document)
    last_error = None

    for round_no in range(1, max_rounds + 1):
        resp = call_with_retry(
            client.chat.completions.create,
            model=settings.deepseek_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2048,
            extra_body={"thinking": {"type": "disabled"}},
        )
        raw = resp.choices[0].message.content

        # token用量
        usage = resp.usage
        if usage:
            logger.info(
                "第 %s 轮 token：prompt=%s completion=%s 缓存命中=%s",
                round_no,
                usage.prompt_tokens,
                usage.completion_tokens,
                getattr(usage, "prompt_cache_hit_tokens", "n/a"),
            )
        if not raw:
            last_error = "模型返回空内容（官方已知问题）"
            logger.warning("第 %s 轮：%s", round_no, last_error)
            continue

        try:
            # 本地校验
            return TenderNotice.model_validate_json(raw)
        except ValidationError as e:
            last_error = str(e)
            logger.warning("第 %s 轮校验失败：%s", round_no, last_error)

            #把错误输出和错误原因一起喂回去，让它自己改
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"上面的 JSON 校验失败，错误如下：\n{last_error}\n"
                    f"请只输出修正后的 JSON，不要任何解释。"
                ),
            })

    raise ValueError(f"{max_rounds} 轮后仍无法得到合法结果：{last_error}")

# 抽取失败要降级
def extract_safe(document: str) -> dict:
    try:
        return extract_with_retry(document).model_dump()
    except Exception as e:
        logger.error("抽取彻底失败：%s", e)
        return {
            "_failed": True,
            "_error": str(e),
            **TenderNotice.empty().model_dump(),
        }
