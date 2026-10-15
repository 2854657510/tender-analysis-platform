
from pathlib import Path

import pymupdf


def parse_pdf(path: str | Path) -> list[dict]:
    """解析 PDF，返回逐页文本。

    返回格式（后面的模块都依赖这个结构，别改）：
        [
            {"page": 1, "text": "第一页的文本..."},
            {"page": 2, "text": "第二页的文本..."},
        ]

    实现提示：
        1. import pymupdf
           ★ 注意：老教程写 `import fitz`，新版已经废弃了（会有 DeprecationWarning）。
             现在统一用 `import pymupdf`，装的就是这个包名。
        2. doc = pymupdf.open(path)
        3. 用 enumerate 遍历每一页，页码从 1 开始
        4. page.get_text() 拿文本
        5. 记得 doc.close()

    注意：
        · 扫描件（图片型 PDF）用 get_text() 会返回【空字符串】。
          这种情况要走 OCR（D2 的 ocr.py），D1 先不做。
        · 空页可以过滤掉，但【别丢页码】—— 用 if text.strip() 判断后 continue，
          而不是重新编号。

    D1 验收：
        · 用一份真实的招投标 PDF，打印出前 3 页的页码和文本前 100 字
        · 打印总页数，和你在 PDF 阅读器里看到的一致
    """
    pages = []

    # with 会自动调 doc.close()，比手写 try/finally 省事，也不会忘
    with pymupdf.open(path) as doc:
        # start=1：页码从 1 开始。这个 page_no 是【真实页码】，不是列表下标
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text().strip()

            # 空页跳过。★ 注意是 continue，不是把后面的页往前挪 —— 页码不能重排
            if not text:
                continue

            pages.append({"page": page_no, "text": text})

    return pages


def extract_metadata(path: str | Path) -> dict:
    """抽取 PDF 的元信息（可选，做不完不影响 D1 验收）。

    返回：{"title": ..., "page_count": ..., "file_name": ...}
    用途：入库时存进 SQLite 文档表，前端列表页要显示。
    """
    path = Path(path)

    with pymupdf.open(path) as doc:
        meta = doc.metadata or {}
        page_count = doc.page_count

    # 网页打印出来的 PDF 常常没有 title，退回用文件名，别让前端显示空白
    title = (meta.get("title") or "").strip() or path.stem

    return {"title": title, "page_count": page_count, "file_name": path.name}
