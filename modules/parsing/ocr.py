
import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

# 判定"扫描件"的阈值：所有页加起来还不到这么多字符，就认为整份是图片。
# ★ 为什么不做"逐页判断"：真实标书常常是"前几页封面是图、后面是文字"的混合体。
#   用整份的平均水平判断更好使；逐页判断会让一份正常文档因为封面是图就整个走 OCR。
SCANNED_TEXT_THRESHOLD = 50

# 渲染 DPI。★ 这是"识别准确率"和"速度"的权衡点：
#   150 以下小字会糊、识别率明显下降；300 以上准确率提升很小但耗时翻倍。
#   200 是本机的实测甜点值。
OCR_DPI = 200

# 单份文档最多 OCR 多少页。
# ★ 必须有限制：OCR 一页要几秒，一份 200 页的扫描标书 = 十几分钟，
#   上传接口早就超时了。超出部分直接放弃并记日志，
#   总比让用户对着一个永远转不完的圈强。
MAX_OCR_PAGES = 50

# 模块级缓存的 OCR 实例。
# ★ 初始化要加载检测+识别两个模型，要好几秒、占几百 MB 内存，
#   必须复用，不能每次调用都新建。
_ocr = None


def is_scanned_pdf(pages: list[dict]) -> bool:
    """判断是不是扫描件：解析出来的文本总体上是不是空的。

    实现：统计所有页的文本总长度，少于阈值（比如 50 字符）就认为是扫描件。

    ★ 注意入参是 parse_pdf 的【结果】而不是文件路径 ——
      这样调用方可以先正常解析一遍，只在"解析出来是空的"时候才判断要不要 OCR，
      省掉"先花力气判断、再回头解析"的重复劳动。
    """
    if not pages:
        # 一页都没解析出来，那多半也是扫描件（或者是个坏文件）
        return True

    total_chars = sum(len(p.get("text", "").strip()) for p in pages)
    return total_chars < SCANNED_TEXT_THRESHOLD


def _get_ocr():
    """懒加载 PaddleOCR 实例。

    ★ 关于 import 顺序的坑（本机实测）：
      这个环境里同时装了 torch（Embedding 要用）和 paddle。
      如果【先 import paddle、再 import torch】，torch 会加载失败：
          OSError: [WinError 127] Error loading "torch\\lib\\shm.dll"
      反过来（torch 先、paddle 后）完全正常。

      本项目里 Embedding 一定先跑（上传时就跑），所以实际顺序天然安全。
      但如果你单独写脚本调 OCR，记得别手动先 import paddleocr。
    """
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR

        logger.info("正在初始化 PaddleOCR（首次要下载模型，稍等）")
        _ocr = PaddleOCR(
            # ★ 下面三个开关默认是【开】的，而它们各自都是一次额外的模型推理：
            #   use_doc_orientation_classify 判断整页有没有转 90 度
            #   use_doc_unwarping            把拍歪的页面拉正
            #   use_textline_orientation     判断每行文字的方向
            #   标书基本都是规规矩矩的打印件，用不上，全关掉能快好几倍。
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # ★★ Windows 上的保命开关，见文件头的实测记录
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR 初始化完成")
    return _ocr


def _pixmap_to_bgr(pix) -> "np.ndarray":
    """把 PyMuPDF 的页面位图转成 OpenCV 风格的 BGR 数组。

    ★ 为什么要转 BGR：PaddleOCR 内部沿用 OpenCV 的约定（BGR），
      而 PyMuPDF 给的是 RGB。不转的话红色和蓝色会互换 ——
      对黑白文字影响不大，但遇到彩色印章、表格线就会出问题。
    """
    import cv2
    import numpy as np

    # pix.samples 是一段裸字节，要按 (高, 宽, 通道数) 重新塑形才能当图像用
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    elif pix.n == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    return arr


def ocr_pdf(path: str | Path) -> list[dict]:
    """对扫描件做 OCR，返回和 parse_pdf 一样的结构。

    返回：[{"page": 1, "text": "..."}, ...]

    ★ 返回结构【必须和 parse_pdf 完全一致】——
      这样下游的 chunker.split_pages 根本不需要知道这份文档是"解析来的"还是
      "OCR 来的"。用一个统一的中间结构把差异挡在边界外，
      是让后面的代码保持简单的关键。
    """
    path = Path(path)
    ocr = _get_ocr()

    pages: list[dict] = []

    with pymupdf.open(path) as doc:
        total = doc.page_count

        # ★ 超出上限的页直接不处理，但清清楚楚地告诉你"还有多少页没做"，
        #   而不是默默丢掉 —— 静默截断是排查问题时最痛的一种设计。
        if total > MAX_OCR_PAGES:
            logger.warning(
                "文档共 %s 页，超过 OCR 上限 %s 页，只处理前 %s 页",
                total, MAX_OCR_PAGES, MAX_OCR_PAGES,
            )

        for page_no, page in enumerate(doc, start=1):
            if page_no > MAX_OCR_PAGES:
                break

            # ① 把这一页渲染成图片。★ 传 dpi 而不是用默认 72 ——
            #   72 DPI 对正文小字来说太糊，识别率会明显掉下来。
            pix = page.get_pixmap(dpi=OCR_DPI)
            image = _pixmap_to_bgr(pix)

            # ② 交给 PaddleOCR。
            #    ★ 3.x 的 predict() 返回的是一个可迭代的结果列表，
            #      每个元素像字典一样用，识别出来的文字在 r["rec_texts"] 里。
            #      （2.x 是 ocr.ocr() 返回 [[[框, (文本, 置信度)], ...]]，写法完全不同）
            result = ocr.predict(image)

            lines: list[str] = []
            for r in result:
                lines.extend(r["rec_texts"])

            # ★ 用换行拼行，不要拼成一大坨：
            #   后面切 chunk 时会按换行分段，保留行结构能让 chunk 边界更合理。
            text = "\n".join(lines).strip()

            if not text:
                # 空白页照样跳过，但【不重新编号】—— 页码是原文的真实页码
                logger.info("第 %s 页 OCR 没识别出内容，跳过", page_no)
                continue

            pages.append({"page": page_no, "text": text})
            logger.info("第 %s/%s 页 OCR 完成（%s 字）", page_no, total, len(text))

    return pages
