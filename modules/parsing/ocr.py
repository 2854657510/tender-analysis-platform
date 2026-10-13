"""OCR 识别 —— D2 任务（可选）。

【什么时候需要它】扫描件 PDF（整页是图片）用 get_text() 会返回空字符串。
    判断方法：parse_pdf 出来的文本全是空的，那基本就是扫描件。

【⚠️ 优先级提醒】
    OCR 是【加分项，不是必需】。paddlepaddle 装起来几百 MB，
    而且首次运行要下模型。
    ★ 如果 D2 卡在装环境上超过 1 小时，【直接跳过】——
      用文本型 PDF 也能跑完整条 RAG 链路，不影响项目主体。
      跳过后在 README 里写明"OCR 未实现"，面试时说明原因即可。
"""
from pathlib import Path


def is_scanned_pdf(pages: list[dict]) -> bool:
    """判断是不是扫描件：解析出来的文本总体上是不是空的。

    实现：统计所有页的文本总长度，少于阈值（比如 50 字符）就认为是扫描件。
    """
    raise NotImplementedError("D2 任务（可选）：判断扫描件")


def ocr_pdf(path: str | Path) -> list[dict]:
    """对扫描件做 OCR，返回和 parse_pdf 一样的结构。

    返回：[{"page": 1, "text": "..."}, ...]

    实现提示：
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch")   # lang="ch" 中文
        # 把 PDF 每页转成图片再识别（fitz 可以 page.get_pixmap()）

    ★ 注意：OCR 很慢（一页要几秒），别在上传接口里同步跑 ——
      要么异步任务，要么限制页数。
    """
    raise NotImplementedError("D2 任务（可选）：实现 OCR")
