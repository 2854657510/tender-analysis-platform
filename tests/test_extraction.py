"""抽取逻辑的单元测试（从项目1复制过来，import 路径已调整）。

【新建文件 2026-09-29】
【为什么这些测试值钱】它们用【假客户端】跑，不花一分钱 API 费用，
但把 4 条关键路径全锁住了。面试时说"我给失败路径写了测试"，比说"我跑通了"有分量得多。

跑法：
    pytest -v
"""
import pytest

import modules.extraction.extract as extract_module
from modules.extraction.extract import extract_safe, extract_with_retry
from modules.extraction.schema import TenderNotice

# 一个完全合法的输出
VALID = (
    '{"project_name": "XX市智慧交通建设项目", "buyer": "XX市交通运输局", '
    '"budget_wan": 1850.0, "deadline": "2026年10月20日", '
    '"qualification": ["具有独立法人资格"], '
    '"contact": {"name": "李工", "phone": "0571-8888"}, "confidence": 0.9}'
)

# 语法合法但【结构不对】—— 这是抽取任务里最常见的失败形态
BAD = (
    '{"project_name": "XX市智慧交通建设项目", "budget_wan": "约1850万", '
    '"qualification": "具有独立法人资格", "confidence": 3}'
)


# ---------------------------------------------------------------------------
# 假客户端：不联网，按脚本依次返回内容，并记录每一轮收到的 messages
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    prompt_tokens = 100
    completion_tokens = 50
    prompt_cache_hit_tokens = 0


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _Completions:
    def __init__(self, script):
        self._script = list(script)
        self.calls = []          # 每一轮的 kwargs，用来断言"喂回去了什么"

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self._script.pop(0))


class _Chat:
    def __init__(self, script):
        self.completions = _Completions(script)


class FakeClient:
    def __init__(self, script):
        self.chat = _Chat(script)

    @property
    def calls(self):
        return self.chat.completions.calls


@pytest.fixture
def fake(monkeypatch):
    """返回一个工厂：传入"剧本"列表，装上假客户端。"""

    def _install(script):
        c = FakeClient(script)
        monkeypatch.setattr(extract_module, "client", c)
        return c

    return _install


# ---------------------------------------------------------------------------
# 路径1：一次成功
# ---------------------------------------------------------------------------
def test_success_first_round(fake):
    c = fake([VALID])
    result = extract_with_retry("随便一段公告")

    assert isinstance(result, TenderNotice)
    assert result.project_name == "XX市智慧交通建设项目"
    assert result.budget_wan == 1850.0
    assert len(c.calls) == 1, "一次成功就不该有第二轮"


# ---------------------------------------------------------------------------
# 路径2 ★ 最重要：校验失败 → 把错误喂回去 → 自愈
# ---------------------------------------------------------------------------
def test_retry_feeds_back_error(fake):
    c = fake([BAD, VALID])
    result = extract_with_retry("随便一段公告")

    assert result.project_name == "XX市智慧交通建设项目"
    assert len(c.calls) == 2, "应该调用两轮"

    second_round_messages = c.calls[1]["messages"]

    # 断言1：模型的错误输出被原样带进上下文了
    assert second_round_messages[-2]["role"] == "assistant"
    assert second_round_messages[-2]["content"] == BAD

    # 断言2：校验错误的具体信息被喂回去了，而且带上了字段名
    feedback = second_round_messages[-1]
    assert feedback["role"] == "user"
    assert "校验失败" in feedback["content"]
    assert "budget_wan" in feedback["content"], (
        "错误反馈里必须带具体字段名，否则模型学不到东西"
    )


# ---------------------------------------------------------------------------
# 路径3：官方已知问题 —— 返回空 content → 重试
# ---------------------------------------------------------------------------
def test_empty_content_triggers_retry(fake):
    c = fake([None, VALID])
    result = extract_with_retry("随便一段公告")

    assert result.project_name == "XX市智慧交通建设项目"
    assert len(c.calls) == 2


# ---------------------------------------------------------------------------
# 路径4：三轮全败 → 降级，不抛异常
# ---------------------------------------------------------------------------
def test_fallback_after_max_rounds(fake):
    c = fake([BAD, BAD, BAD])
    result = extract_safe("随便一段公告")

    assert result["_failed"] is True
    assert "_error" in result
    # 降级结果必须【结构一致】—— 下游代码才能继续跑
    assert result["project_name"] is None
    assert result["qualification"] == []
    assert result["contact"] == {"name": None, "phone": None}
    assert len(c.calls) == 3, "max_rounds=3，不能无限重试"


# ---------------------------------------------------------------------------
# 请求参数断言：关思考模式 + JSON Output + 别截断
# ---------------------------------------------------------------------------
def test_request_params(fake):
    c = fake([VALID])
    extract_with_retry("随便一段公告")

    kw = c.calls[0]
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["max_tokens"] >= 1024, "max_tokens 太小会把 JSON 截断"
    assert kw["extra_body"] == {"thinking": {"type": "disabled"}}, (
        "抽取任务要关思考模式，否则 temperature 不生效且浪费 token"
    )


# ---------------------------------------------------------------------------
# HTTP 层测试
#
# 【说明】项目1 里这部分测的是它自己的接口（/api/extract）。
# 项目2 的接口不一样，等 D12 写完 documents_api / query_api 之后，
# 照着这里的写法自己补 —— 用 TestClient + monkeypatch 替换真实调用。
# ---------------------------------------------------------------------------
