
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
