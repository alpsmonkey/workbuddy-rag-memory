"""
WorkBuddy RAG HTTP 服务端（v0.2.3）

把 Memory 暴露成 HTTP API，让任何应用都能调用检索。

端点：
  GET  /health                    健康检查
  GET  /stats                     索引统计
  POST /search                    检索
  POST /add                       写入单条记忆
  POST /batch_search              批量检索（一次请求多个 query）

启动：
  python scripts/server.py
  python scripts/server.py --host 0.0.0.0 --port 8000
  python scripts/server.py --reload  # 开发模式

API 文档：访问 http://localhost:8000/docs
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 默认离线（防止启动时连 HuggingFace）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.memory import Memory, get_default_embedder


# ============================================================================
# 配置
# ============================================================================

class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    index_dir: Optional[str] = None
    log_level: str = "INFO"
    enable_rerank: bool = False
    enable_hyde: bool = False


# ============================================================================
# Pydantic models（请求 / 响应 schema）
# ============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., description="检索关键词或问题", min_length=1)
    top_k: int = Field(5, ge=1, le=50, description="返回条数")
    project: Optional[str] = Field(None, description="按项目过滤")
    source: Optional[str] = Field(None, description="按 source 过滤")
    rerank: bool = Field(False, description="启用 BGE Reranker")
    use_hyde: bool = Field(False, description="启用 HyDE Query 改写")
    candidates: int = Field(20, ge=5, le=100, description="候选数（RRF 前 N）")


class SearchHit(BaseModel):
    id: str
    text: str
    score: float
    project: Optional[str] = None
    source: Optional[str] = None
    ts: Optional[str] = None
    confidence: float = 0.0
    rerank_score: Optional[float] = None
    access_count: int = 0


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
    count: int
    duration_ms: float


class AddRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: Optional[str] = Field(None, description="记忆来源（如文件名）")
    project: Optional[str] = Field(None, description="强制 project 标签")


class AddResponse(BaseModel):
    decision: str
    reason: str = ""
    existing_id: Optional[str] = None
    similarity: float = 0.0


class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1, max_length=50)
    top_k: int = Field(5, ge=1, le=20)
    rerank: bool = False
    use_hyde: bool = False


class BatchSearchResponse(BaseModel):
    results: List[SearchResponse]


class StatsResponse(BaseModel):
    total_chunks: int
    total_projects: int
    backend: str
    embedding_dim: int
    index_dir: str


class HealthResponse(BaseModel):
    status: str
    version: str
    index_dir: str
    backend: str


# ============================================================================
# 全局 Memory 实例（启动时创建）
# ============================================================================

memory: Optional[Memory] = None
reranker_instance = None
hyde_instance = None


def create_app(
    memory_instance: Memory = None,
    enable_rerank: bool = False,
    enable_hyde: bool = False,
) -> FastAPI:
    """工厂函数：创建 FastAPI app（方便测试）"""
    global memory, reranker_instance, hyde_instance

    memory = memory_instance or Memory()

    if enable_rerank:
        try:
            from src.reranker import Reranker
            reranker_instance = Reranker()
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("Reranker 加载失败: %s", e)

    if enable_hyde:
        from src.hyde import make_default_hyde
        hyde_instance = make_default_hyde()

    app = FastAPI(
        title="WorkBuddy RAG Memory API",
        version="0.2.3",
        description="RAG 增强记忆系统 HTTP 接口",
    )

    # CORS（允许浏览器跨域调用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # 端点
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            version="0.2.3",
            index_dir=str(memory.index_dir),
            backend=memory.embedder.backend,
        )

    @app.get("/stats", response_model=StatsResponse)
    def stats():
        s = memory.indexer.stats()
        return StatsResponse(
            total_chunks=s.get("total", 0),
            total_projects=len(s.get("by_project", {})),
            backend=memory.embedder.backend,
            embedding_dim=memory.embedder.dim,
            index_dir=str(memory.index_dir),
        )

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        import time
        t0 = time.time()
        try:
            results = memory.search(
                req.query,
                top_k=req.top_k,
                project=req.project,
                source=req.source,
                rerank=req.rerank or (reranker_instance is not None and req.rerank),
                use_hyde=req.use_hyde,
                candidates=req.candidates,
            )
        except (RuntimeError, ValueError, OSError) as e:
            raise HTTPException(status_code=500, detail=f"search failed: {e}")

        hits = [
            SearchHit(
                id=r.id,
                text=r.text,
                score=r.score,
                project=r.project or None,
                source=r.source or None,
                ts=r.ts or None,
                confidence=r.confidence,
                rerank_score=r.rerank_score,
                access_count=r.access_count,
            )
            for r in results
        ]

        return SearchResponse(
            query=req.query,
            hits=hits,
            count=len(hits),
            duration_ms=(time.time() - t0) * 1000,
        )

    @app.post("/batch_search", response_model=BatchSearchResponse)
    def batch_search(req: BatchSearchRequest):
        import time
        results = []
        for q in req.queries:
            t0 = time.time()
            hits = memory.search(
                q, top_k=req.top_k, rerank=req.rerank, use_hyde=req.use_hyde,
            )
            results.append(SearchResponse(
                query=q,
                hits=[
                    SearchHit(
                        id=r.id, text=r.text, score=r.score,
                        project=r.project or None, source=r.source or None,
                        ts=r.ts or None, confidence=r.confidence,
                        rerank_score=r.rerank_score, access_count=r.access_count,
                    )
                    for r in hits
                ],
                count=len(hits),
                duration_ms=(time.time() - t0) * 1000,
            ))
        return BatchSearchResponse(results=results)

    @app.post("/add", response_model=AddResponse)
    def add(req: AddRequest):
        try:
            result = memory.add(req.text, source=req.source)
        except (RuntimeError, ValueError, OSError) as e:
            raise HTTPException(status_code=500, detail=f"add failed: {e}")
        return AddResponse(
            decision=result.decision.value if hasattr(result.decision, "value") else str(result.decision),
            reason=result.reason,
            existing_id=result.existing_id,
            similarity=result.similarity,
        )

    return app


# ============================================================================
# 入口
# ============================================================================

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="WorkBuddy RAG HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="绑定 host（默认 localhost）")
    parser.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    parser.add_argument("--index-dir", default=None, help="索引目录（默认 ~/.workbuddy/rag-index）")
    parser.add_argument("--rerank", action="store_true", help="启用 BGE Reranker")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE Query 改写")
    parser.add_argument("--reload", action="store_true", help="开发模式：自动重载")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # 配置 logging
    from scripts._logging import setup_logging
    setup_logging(level=args.log_level)

    # 创建 Memory
    if args.index_dir:
        mem = Memory(index_dir=args.index_dir)
    else:
        mem = Memory()

    # 创建 app
    app = create_app(mem, enable_rerank=args.rerank, enable_hyde=args.hyde)

    # 启动 uvicorn
    import uvicorn
    logger.info("🚀 WorkBuddy RAG server starting on http://%s:%d", args.host, args.port)
    logger.info("   文档: http://%s:%d/docs", args.host, args.port)
    logger.info("   索引: %s", mem.index_dir)
    logger.info("   后端: %s", mem.embedder.backend)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()