# 智能标书分析平台

> 上传标书 PDF → 自动解析入库 → 自然语言问答（带出处）→ Agent 自动完成对比分析

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-orange)]()



---

## 解决什么问题

一份标书几十上百页，想找"这个项目的资格要求是什么"，只能 Ctrl+F 或者从头翻。

**这个平台把标书变成可检索的知识库**：
- 用 RAG 做精准问答，答案带出处，可点回原文
- 复杂的对比分析交给 Agent 拆解成多步自动完成

**那个项目解决了"单篇文档转结构化"，但处理不了跨文档检索** ——
这就是本项目引入 RAG 的原因。

## 架构

```
                    ┌──────────────────────────────────────┐
                    │      Streamlit 前端（多页面）          │
                    │  ① 上传文档  ② 知识库问答  ③ 报告分析  │
                    └────────────────┬─────────────────────┘
                                     │ HTTP
                    ┌────────────────▼─────────────────────┐
                    │         FastAPI 后端                   │
                    └────────────────┬─────────────────────┘
                                     │
      ┌──────────────────────────────┼──────────────────────────────┐
      │                              │                              │
┌─────▼──────────┐        ┌──────────▼──────────┐        ┌──────────▼─────────┐
│  ① 文档处理层   │        │   ② 检索层           │        │  ③ Agent 编排层    │
│ PDF 解析 / OCR  │        │ 向量 + BM25 + RRF    │        │ LangGraph 状态图    │
│ 切分 / Embedding│        │ + Rerank + 改写      │        │ 手写循环 + MCP 工具 │
└────────────────┘        └─────────────────────┘        └────────────────────┘
      │                              │                              │
      └──────────────────────────────┼──────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │  Chroma（向量）· SQLite（文档元数据）  │
                    └──────────────────────────────────────┘
```

## 技术栈

| 层 | 选型 | 为什么 |
|---|---|---|
| 后端 | FastAPI + Pydantic | 自动文档、类型安全、原生 async |
| 文档解析 | PyMuPDF（+ PaddleOCR） | 快且纯 Python；扫描件走 OCR |
| 切分 | 递归切分 | 尊重段落边界，比固定长度效果好 |
| Embedding | BGE-small-zh | **中文场景必须用中文模型** |
| 向量库 | Chroma | 能嵌进程、零运维。数据量没到需要 Milvus 的级别 |
| 关键词检索 | rank_bm25 + jieba | 补向量检索对专有名词/编号不敏感的短板 |
| 融合 | RRF | 分数不可比，按排名融合天然消除量纲差异 |
| 重排 | BGE-reranker | 粗筛（召回）→ 精排（准确率） |
| 编排 | LangGraph | 状态图可控、可画、可维护 |
| 工具协议 | MCP | 写一次，任何支持 MCP 的客户端都能用 |
| 前端 | Streamlit 多页面 | 快速出界面，专注后端能力 |

## 快速开始

```bash
# 1. 环境（本机用的是 langchain1.2 这个 conda 环境）
conda activate langchain1.2
pip install -r requirements.txt

# 2. 配密钥
cp .env.example .env      # 然后编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 开工前自检（会告诉你哪些还没实现）
python scripts/check_env.py

# 4. 启动后端（终端1）
uvicorn app.main:app --reload

# 5. 启动前端（终端2）
streamlit run frontend/app.py
```

- 前端界面：http://localhost:8501
- 接口文档：http://127.0.0.1:8000/docs

## 目录结构

```
tender-analysis-platform/
├── app/
│   ├── config.py               # 配置（绝对路径读 .env）
│   ├── db.py                   # SQLite 文档元数据 ✅
│   ├── main.py                 # FastAPI 入口 ✅
│   ├── api/
│   │   ├── documents_api.py    # 上传/列表/删除
│   │   ├── query_api.py        # RAG 问答
│   │   └── analysis_api.py     # Agent 分析
│   └── agent/                  # Agent 编排
│       ├── tools.py            # 工具定义
│       ├── loop.py             # ★ 手写 Agent 循环（D6）
│       ├── graph.py            # LangGraph 状态图（D7）
│       └── mcp_server.py       # MCP Server（D8）
├── modules/
│   ├── extraction/             # ★ 从项目1复用：结构化抽取
│   ├── parsing/                # PDF 解析 / 切分 / OCR
│   ├── retrieval/              # 向量 / BM25 / RRF / Rerank / 改写
│   └── llm/client.py           # 模型调用封装 ✅
├── frontend/                   # Streamlit 多页面
├── prompts/                    # 提示词模板
├── scripts/
│   ├── check_env.py            # 开工自检 + 进度追踪 ✅
│   └── ingest.py               # 批量入库
├── tests/
└── data/                       # 上传的文件（gitignore）
```

## 开发进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| **D1** | PDF 解析 + 切分 | ✅ |
| **D2** | OCR + Embedding + 向量库入库 | ✅ |
| **D3** | 最小 RAG 链路 | ✅ |
| **D4** | 混合检索（向量 + BM25 + RRF） | ✅ |
| **D5** | Rerank + Query Rewrite | ✅ |
| **D6** | 手写 Agent 循环 | ✅ |
| **D7** | LangGraph 编排 | ✅ |
| **D8** | MCP Server | ✅ |
| **D9** | Dify 低代码对照 | ⬜ |
| **D10** | 评测体系 | ⬜ |
| **D11** | Memory + 缓存 + 限流 | 
| **D12** | 前端整合 | ✅ |
| **D13** | Docker 部署 | ⬜ |

跑 `python scripts/check_env.py` 可以看到实时的实现进度。

## 已知限制

- Rerank 走的是"一次调用给所有候选打分"，模型偶发失败时**自动降级为不重排**（有日志），不会让接口报错
- 缓存和限流都是**进程内**实现：`uvicorn --workers 2` 时每个 worker 各算各的，生产要换 Redis
- OCR 已实现（PaddleOCR 3.x），但单份文档默认最多 OCR 50 页（`modules/parsing/ocr.py` 里的 `MAX_OCR_PAGES`）


