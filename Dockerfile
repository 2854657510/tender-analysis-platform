# =============================================================================
# 智能标书分析平台 —— 后端 / 前端共用镜像
#
# 为什么共用一个镜像：两个服务的 Python 依赖几乎完全一样（前端只多一个 streamlit，
# 而后端也装了它）。分两个镜像等于把 torch 那一大坨装两遍 —— 镜像体积翻倍。
# 用同一个镜像、compose 里跑不同 command，是这类项目的标准做法。
#
# 构建：docker compose build
# 大小预估：约 1.5~2 GB（大头是 torch 和 chromadb，已经用了 CPU 版 torch 压过一轮）
# =============================================================================
FROM python:3.13-slim

# ---------------------------------------------------------------------------
# 环境变量
#
# ★ PYTHONUNBUFFERED=1 —— 这一行是【容器日志能不能实时看到】的开关。
#   不加的话 Python 会缓冲 stdout，docker logs 要等缓冲区满了才吐东西，
#   你会看到「服务明明在跑，日志却是一片空白」，排查问题时非常误导人。
#
# ★ PYTHONDONTWRITEBYTECODE=1 —— 不生成 .pyc，容器里的代码是只读的，
#   写了也没意义，还平白多出一堆文件。
#
# ★ HF_ENDPOINT —— 和 app/config.py 里那行是同一个目的：国内直连
#   huggingface.co 会超时重试，导致「模型早就下好了，启动还是卡两分钟」。
#   构建阶段（下面预下载模型）也需要它，所以在这里就设上。
#   放在 ENV 里而不是只靠代码，是为了让构建期的 pip/下载也吃到这个变量。
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    PYTHONIOENCODING=utf-8

WORKDIR /app

# ---------------------------------------------------------------------------
# 系统依赖
#
# ★ curl 只是为了 compose 里的 healthcheck 用（探 /health）。
#   装完立刻删 apt 缓存，同一个 RUN 里做 —— 分开写的话，
#   上一层的缓存文件会永久留在镜像里。
#
# ★★ 必须先换 apt 源：python:3.13-slim 默认指向 deb.debian.org，
#    国内服务器拉它的 Packages 索引（9.6MB）会直接卡死 ——
#    实测就是「日志几分钟一动不动」，看起来像构建挂了。
#    换成本地镜像站之后这一步是秒级的。
#    这里用 mirrors.cloud.tencent.com（腾讯云内网直达）；
#    换别的云/自建机时改成对应镜像站即可（阿里云 mirrors.aliyun.com、
#    清华 mirrors.tuna.tsinghua.edu.cn 都行）。
# ---------------------------------------------------------------------------
RUN sed -i 's|http://deb.debian.org|https://mirrors.cloud.tencent.com|g' \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# ① 先装 CPU 版 torch
#
# ★★ 为什么单独装、且必须指定 --index-url：
#    PyPI 上的 torch 默认是 CUDA 版，光它自己就 1.5GB+，
#    CPU 版只有 200MB 左右。服务器没有 GPU，装 CUDA 版纯属白烧一倍镜像体积
#    和十几分钟构建时间。PyTorch 官方为此单独提供了 cpu 源。
#
# ★ 放在装 requirements 之前：torch 装好之后，
#   requirements 里的 sentence-transformers 就不会再拉一份 CUDA torch 回来。
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir torch==2.14.1 \
        --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# ② 再装其余依赖
#
# ★ 先 COPY requirements.txt 再 COPY 代码，是【有意为之的顺序】：
#   Docker 逐层缓存，只要 requirements.txt 没变，改了业务代码重建时
#   这一层直接命中缓存、跳过几分钟的安装。反过来的话每次都重装一遍。
#
# ★ PIP_INDEX：默认走清华镜像源。国内服务器直连 pypi.org 经常只有几十 KB/s，
#   装这几个包能从十几分钟缩到一两分钟。
#   在国外部署时可以用 --build-arg PIP_INDEX_URL=https://pypi.org/simple 覆盖。
# ---------------------------------------------------------------------------
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} -r requirements.txt

# ---------------------------------------------------------------------------
# ③ 内置中文 Embedding 模型（BAAI/bge-small-zh-v1.5，约 95MB）
#
# ★★ 为什么这里是 COPY 而不是 RUN pip/huggingface 下载：
#    实测这台服务器【访问 hf-mirror.com 会 TLS 握手超时】（huggingface.co
#    直连更不通），在构建期下载会把构建卡死在那里 —— 日志表现为反复
#    "handshake operation timed out / Retrying"，看起来像构建挂了。
#
#    改成：构建之前先在【宿主机】把模型下好放进 models/ 目录，构建时直接拷进镜像。
#    下载源用 ModelScope（阿里，国内可达，实测 6MB/s），命令：
#      mkdir -p models && cd models
#      BASE='https://www.modelscope.cn/api/v1/models/BAAI/bge-small-zh-v1.5/repo?Revision=master&FilePath='
#      for f in config.json config_sentence_transformers.json model.safetensors \
#               modules.json sentence_bert_config.json special_tokens_map.json \
#               tokenizer.json tokenizer_config.json vocab.txt; do
#          curl -sL -o "bge-small-zh-v1.5_$f" "${BASE}${f}"
#      done
#      # 再补一个子目录文件：1_Pooling/config.json（FilePath 里的 / 要编码成 %2F）
#
# ★ 模型放进镜像而不是运行时下载：第一次上传/提问就不会卡在下载上。
#   代码通过 EMBEDDING_MODEL 环境变量指向这个目录（见 docker-compose.yml）。
# ---------------------------------------------------------------------------
COPY models/ /app/models/

# ---------------------------------------------------------------------------
# ④ 最后才拷代码
# ★ 代码改动最频繁，放最后能让上面的依赖层和模型层最大限度复用缓存。
# ---------------------------------------------------------------------------
COPY . .

# 后端端口（前端 8501 由 compose 里覆盖 command 时另开）
EXPOSE 8000

# ★ --host 0.0.0.0 是【容器外能访问】的关键：
#   默认只绑 127.0.0.1，那是容器内网回的地址，宿主机根本连不上 ——
#   表现就是「容器明明是 Up，浏览器就是打不开」。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
