
import json
import os
from typing import Iterator
from urllib.parse import quote

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

    ★ timeout 要设大一点（默认 5 分钟）—— 解析 + Embedding 很慢，
      requests 默认没有超时上限的话会一直挂着，而设成 5 秒又必然超时。
    """
    # ★ multipart 上传的标准写法：{"字段名": (文件名, 字节流, MIME类型)}
    #   后端 FastAPI 那边的参数名是 file，这里必须对得上。
    files = {"file": (file_name, file_bytes, "application/pdf")}

    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/documents/upload", files=files, timeout=timeout
        )
    except requests.exceptions.ConnectionError as e:
        raise BackendDown(str(e)) from e

    if r.status_code != 200:
        # ★ 把后端的 detail 尽量带出来 —— 用户最需要知道的是"为什么失败"
        #   （是扫描件？超体积？），而不是一个干巴巴的 500。
        try:
            detail = r.json().get("detail", r.text[:300])
        except Exception:
            detail = r.text[:300]
        raise BackendError(r.status_code, detail)

    return r.json()


def delete_document(base_url: str, file_name: str, timeout: int = 60) -> dict:
    """删除一份文档（后端会同时清向量库和 SQLite）。"""
    try:
        # ★ 文件名里有中文和空格，要 quote 转义，
        #   直接拼进 URL 会 400 或者匹配不到路由。
        r = requests.delete(
            f"{base_url.rstrip('/')}/api/documents/{quote(file_name, safe='')}", timeout=timeout
        )
    except requests.exceptions.ConnectionError as e:
        raise BackendDown(str(e)) from e
    if r.status_code != 200:
        raise BackendError(r.status_code, r.text[:300])
    return r.json()


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
