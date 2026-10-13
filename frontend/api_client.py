"""前端接口调用层 —— 所有网络请求都收在这里。

【设计原则】前端不 import 后端的任何代码，只通过 HTTP 调用。
    好处：后端挂了前端照样能启动，只提示"连不上后端"；两者可分开部署。

【从项目1复制】后端地址读环境变量的做法沿用：
    本地不设 BACKEND_URL（用默认值），Docker 里由 compose 注入 http://api:8000
"""
import json
import os
from typing import Iterator

import requests

DEFAULT_BASE_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")


class BackendDown(Exception):
    """连不上后端。"""


class BackendError(Exception):
    """后端返回了错误状态码。"""

    def __init__(self, status_code: int, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}")


def check_health(base_url: str, timeout: int = 5) -> bool:
    """探测后端是否活着。侧边栏那个绿点用。"""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("status") is True
    except Exception:
        return False


def upload_document(base_url: str, file_bytes: bytes, file_name: str, timeout: int = 300) -> dict:
    """上传 PDF。

    实现提示：
        files = {"file": (file_name, file_bytes, "application/pdf")}
        r = requests.post(f"{base_url}/api/documents/upload", files=files, timeout=timeout)
    ★ timeout 要设大一点 —— 解析 + Embedding 很慢，默认 5 秒必超时。
    """
    raise NotImplementedError("D12 任务：实现文件上传")


def list_documents(base_url: str) -> list[dict]:
    """列出已上传的文档。这个【可以直接用】。"""
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/documents", timeout=30)
    except requests.exceptions.ConnectionError as e:
        raise BackendDown(str(e)) from e
    if r.status_code != 200:
        raise BackendError(r.status_code, r.text[:300])
    return r.json().get("documents", [])


def query(base_url: str, question: str, timeout: int = 180) -> dict:
    """RAG 问答。

    返回：{"answer": "...", "sources": [...]}
    """
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/query", json={"question": question}, timeout=timeout
        )
    except requests.exceptions.ConnectionError as e:
        raise BackendDown(str(e)) from e
    if r.status_code != 200:
        raise BackendError(r.status_code, r.text[:300])
    return r.json()


def analyze(base_url: str, question: str, timeout: int = 300) -> dict:
    """Agent 分析（返回答案 + 执行步骤）。"""
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/analysis", json={"question": question}, timeout=timeout
        )
    except requests.exceptions.ConnectionError as e:
        raise BackendDown(str(e)) from e
    if r.status_code != 200:
        raise BackendError(r.status_code, r.text[:300])
    return r.json()
