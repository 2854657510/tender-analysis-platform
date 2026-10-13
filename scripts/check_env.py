"""开工前自检 + 进度追踪。

用法：
    cd "C:\\Users\\Firebat\\Desktop\\tender-analysis-platform"
    conda activate langchain1.2
    $env:PYTHONIOENCODING="utf-8"
    python scripts/check_env.py

【为什么需要它】
1. 每次开工前跑一次，5 秒确认环境没问题（项目1踩过的坑这里一次性排掉）
2. 它能扫描出哪些模块还是 NotImplementedError —— 相当于进度条
"""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("=" * 62)
print("  智能标书分析平台 · 开工自检")
print("=" * 62)

# ---------------------------------------------------------------------------
# 1. 解释器
# ---------------------------------------------------------------------------
print(f"\n[1] Python: {sys.version.split()[0]}")
print(f"    解释器: {sys.executable}")
if "langchain1.2" not in sys.executable:
    print("    ⚠️  不在 langchain1.2 环境里！先 conda activate langchain1.2")
else:
    print("    ✅ 环境正确")

# ---------------------------------------------------------------------------
# 2. 配置（项目1踩过的坑：.env 读不到会静默掉回默认值）
# ---------------------------------------------------------------------------
print("\n[2] 配置加载")
try:
    from app.config import BASE_DIR, settings

    print(f"    BASE_DIR = {BASE_DIR}")
    print(f"    base_url = {settings.deepseek_base_url}")
    print(f"    model    = {settings.deepseek_model}")

    assert settings.deepseek_base_url.startswith("http"), "base_url 不是完整 URL！"
    assert settings.deepseek_base_url != "https://deepseek.com", "掉回默认域名了，.env 没读到"
    assert (BASE_DIR / ".env").is_file(), f"{BASE_DIR / '.env'} 不存在"
    print("    ✅ .env 读取正常")
except Exception as e:
    print(f"    ❌ {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# 3. 依赖装了没（按天数分组，缺了会提示你哪天需要）
# ---------------------------------------------------------------------------
print("\n[3] 依赖检查")
DEPS = [
    ("fastapi", "项目1沿用", True),
    ("pydantic_settings", "项目1沿用", True),
    ("openai", "项目1沿用", True),
    ("jinja2", "项目1沿用", True),
    ("streamlit", "项目1沿用", True),
    ("pymupdf", "D1 PDF解析", False),
    ("langchain_text_splitters", "D1 递归切分", False),
    ("chromadb", "D2 向量库", False),
    ("sentence_transformers", "D2 Embedding", False),
    ("rank_bm25", "D4 关键词检索", False),
    ("jieba", "D4 中文分词", False),
    ("langgraph", "D7 Agent编排", False),
    ("mcp", "D8 MCP Server", False),
]
for mod, purpose, required_now in DEPS:
    try:
        importlib.import_module(mod)
        print(f"    ✅ {mod:28} ({purpose})")
    except ImportError:
        mark = "❌" if required_now else "○ "
        note = "  ← 现在就该装" if required_now else "  ← 到那天再装"
        print(f"    {mark} {mod:28} ({purpose}){note}")

# ---------------------------------------------------------------------------
# 4. ★ 进度追踪：哪些模块还是空的
# ---------------------------------------------------------------------------
print("\n[4] 实现进度（扫 NotImplementedError）")
TODO_UNITS = [
    ("modules/parsing/pdf_parser.py", "D1", "PDF 解析"),
    ("modules/parsing/chunker.py", "D1", "文本切分"),
    ("modules/parsing/ocr.py", "D2", "OCR（可选）"),
    ("modules/retrieval/vector_store.py", "D2", "向量库"),
    ("modules/retrieval/rag.py", "D3", "RAG 主链路"),
    ("modules/retrieval/bm25_search.py", "D4", "BM25"),
    ("modules/retrieval/hybrid.py", "D4", "混合检索"),
    ("modules/retrieval/rerank.py", "D5", "Rerank"),
    ("modules/retrieval/query_rewrite.py", "D5", "查询改写"),
    ("app/agent/loop.py", "D6", "手写 Agent 循环"),
    ("app/agent/graph.py", "D7", "LangGraph 编排"),
    ("app/agent/mcp_server.py", "D8", "MCP Server"),
    ("app/api/documents_api.py", "D12", "文档接口"),
    ("app/api/query_api.py", "D12", "问答接口"),
    ("app/api/analysis_api.py", "D12", "分析接口"),
]

done, todo = [], []
for rel, day, name in TODO_UNITS:
    p = ROOT / rel
    if not p.exists():
        todo.append((day, name, "文件不存在"))
        continue
    text = p.read_text(encoding="utf-8")
    n = text.count("NotImplementedError")
    (todo if n else done).append((day, name, n))

for day, name, n in todo:
    print(f"    ○  {day:4} {name:16} 待实现（{n} 处）")
for day, name, _ in done:
    print(f"    ✅ {day:4} {name:16} 已完成")

print(f"\n    进度：{len(done)} / {len(done) + len(todo)}")
if todo:
    nxt = todo[0]
    print(f"    👉 下一件事：{nxt[0]} · {nxt[1]}")

print("\n" + "=" * 62)
