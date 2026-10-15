
import json
import logging

from modules.llm.client import chat

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你是一个检索优化助手。把用户的问题改写成更适合在招投标文档中检索的形式。

规则：
1. 把口语化表达换成招投标领域的规范术语
2. 保留原问题的所有关键信息，不要扩大或缩小范围
3. 输出 1 条改写后的问题，只输出问题本身，不要解释

用户问题：{question}
改写后："""


def rewrite(question: str) -> str:
    """把问题改写成更适合检索的形式。

    实现步骤：
        1. 用 REWRITE_PROMPT 格式化
        2. chat([{"role": "user", "content": prompt}])
        3. 返回模型输出（记得 strip()，去掉可能的空白）

    ★ 注意：改写【可能失败或跑偏】。如果改写后的结果和原问题差异过大，
      要考虑退回用原问题检索 —— 别让改写把检索带沟里。
    """
    try:
        prompt = REWRITE_PROMPT.format(question=question)

        # ★ temperature=0：改写是"加工"不是"创作"，要的是稳定、可复现，
        #   同一个问题每次都该改写出差不多的结果
        rewritten = chat([{"role": "user", "content": prompt}], temperature=0.0).strip()

        # ★ 长度检查：模型有时会"太热心"，把问题扩写成一段话，
        #   或者干脆把规则也复述一遍。超过原问题 3 倍就认为跑偏了，退回原问题。
        if not rewritten or len(rewritten) > len(question) * 3:
            logger.warning("改写结果异常（长度 %s → %s），退回原问题", len(question), len(rewritten))
            return question

        logger.info("查询改写：%s → %s", question, rewritten)
        return rewritten

    except Exception as e:
        # ★ 降级：改写只是"提速手段"，不是必需品。
        #   挂了就用原问题检索 —— 效果差点，但功能不能断。
        logger.warning("查询改写失败，使用原问题：%s", e)
        return question


def multi_query(question: str, n: int = 3) -> list[str]:
    """生成多个不同角度的问法（可选，做不完不影响 D5 验收）。

    用途：分别检索后合并结果，提高召回率。
    """
    prompt = f"""请从不同角度把下面的问题改写成 {n} 个不同的检索问法，用于在招投标文档中检索。

要求：
1. 每个问法侧重点不同（比如分别从金额、时间、资格要求、联系方式等角度切入）
2. 保持原问题的核心意图不变
3. 只输出 JSON 数组，格式为 ["问法1", "问法2", "问法3"]，不要任何解释

原问题：{question}"""

    try:
        raw = chat([{"role": "user", "content": prompt}], json_mode=True, max_tokens=512)
        data = json.loads(raw)

        # 兼容 {"queries": [...]} 和直接给数组两种形态
        queries = data.get("queries", []) if isinstance(data, dict) else data
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

        # ★ 一定要把【原问题】也放进列表：改写出来的问法再合理，
        #   也是"二手"的；原问题才是最忠实的意图表达。
        return [question] + queries[:n]

    except Exception as e:
        logger.warning("多查询扩展失败，只用原问题：%s", e)
        return [question]
