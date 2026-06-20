FROM python:3.13-slim

LABEL maintainer="peng@example.com"
LABEL description="WorkBuddy RAG 增强记忆系统"
LABEL version="0.2.0"

# 系统依赖（lancedb 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（Docker 层缓存）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir watchdog pytest

# 复制源码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY data/ ./data/
COPY pyproject.toml Makefile ./
COPY download_bge_m3.py download_bge_reranker.py ./

# 下载模型（镜像构建期一次性下，运行时无需联网）
RUN python download_bge_m3.py && python download_bge_reranker.py

# 预创建默认索引目录
RUN mkdir -p /root/.workbuddy/rag-index /root/.workbuddy/memory

# 默认命令：跑健康检查
CMD ["python", "-m", "scripts.health"]