"""文本切分（chunking）—— D1 任务。

【要做什么】把长文本切成一个个 chunk，每个 chunk 记住它来自第几页。

【为什么必须切分】模型上下文窗口装不下整份标书；
而且检索的最小单位就是 chunk —— **chunk 的质量直接决定 RAG 的上限**。

【三种切分策略，你要能说出选了哪种、为什么】
    ① 固定长度切分    —— 简单粗暴，会在句子中间断开
    ② 递归切分        —— 按"段落 → 句子 → 词"的优先级切，尽量保持语义完整（推荐）
    ③ 语义切分        —— 用 Embedding 判断语义边界，最准但最慢最贵

【关键参数】两个都要你调出来：
    chunk_size    每个 chunk 多大
    chunk_overlap 相邻 chunk 重叠多少（防止答案正好落在切口上被切断）
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

# 分隔符按"优先级从高到低"排：先试段落，不行退到句子，再不行退到短句，最后才硬切。
# ★ 中文文档【必须】显式传这个 —— 默认分隔符是 ["\n\n", "\n", " ", ""]，
#   对中文等于只有换行可用，一句话超过 chunk_size 就会被硬切在字中间。
SEPARATORS = ["\n\n", "\n", "。", "；", "，", ""]

# 在模块级建一次就够了。切分器是无状态的，不用每次调用都新建。
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,       # 每块多大（app/config.py: 500）
    chunk_overlap=settings.chunk_overlap,  # 相邻重叠多少（app/config.py: 50）
    separators=SEPARATORS,
)


def split_pages(pages: list[dict]) -> list[dict]:
    """把 parse_pdf() 的输出切成 chunk。

    输入：
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]

    输出（结构固定，后面的检索模块依赖它）：
        [
            {"text": "chunk 正文", "page": 1, "chunk_index": 0},
            {"text": "chunk 正文", "page": 1, "chunk_index": 1},
            ...
        ]

    ★ 三个必须做到的点：
        1. 【每个 chunk 都要带 page】—— 回答标出处靠它
        2. 【chunk_index 全局递增】，不是每页重新从 0 开始 —— 便于调试时定位
        3. 【chunk 之间要有 overlap】—— 否则答案被切口切断就检索不到了

    实现提示：
        · 可以自己写（按段落 split，超长再按句号切）
        · 也可以用 LangChain 的 RecursiveCharacterTextSplitter（推荐，你已学 LangChain）
        · 中文分隔符要加：["\\n\\n", "\\n", "。", "；", "，", ""]

    D1 验收：
        · 切完后打印 chunk 总数
        · 随机抽 3 个 chunk，能看到它来自第几页
        · 检查相邻 chunk 确实有重叠内容
        · ★ 能说出你选的 chunk_size 和理由（面试会问）
    """
    chunks = []
    index = 0  # ★ 全局编号，跨页连续 —— 不是每页从 0 重新开始

    for page in pages:
        for piece in split_text(page["text"]):
            chunks.append({
                "text": piece,
                "page": page["page"],  # ★ 页码从 parse_pdf 的结果原样带过来，一个字都不能改
                "chunk_index": index,
            })
            index += 1

    return chunks


def split_text(text: str) -> list[str]:
    """单页文本的切分（split_pages 内部会调用它）。

    单独抽出来是为了好测试 —— 传一段文本进去，看切出来对不对。

    返回 list[str]（只有文本，没有页码和编号）：
        页码和编号是 split_pages 的职责。这个函数只回答"怎么切"这一个问题，
        职责单一，才好在 REPL 里喂一段文本单独测。
    """
    pieces = _SPLITTER.split_text(text)

    # 递归切分最后一级会掉出空串/纯空白碎片，过滤掉
    return [p.strip() for p in pieces if p.strip()]
