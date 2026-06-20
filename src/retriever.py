"""
混合检索: 向量 + BM25 + 元数据过滤，三路 RRF 融合

阶段 2 增强:
- 时间衰减重排: score = rrf × exp(-Δt_days / τ) × log(1 + access_count)
- 默认 RRF_K=30 (阶段 1 是 60，区分度太低)
"""
from __future__ import annotations
import json
import math
import re
import logging
import sqlite3
from datetime import datetime, date
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    from .indexer import Indexer
    from .embedder import Embedder
    from .reranker import Reranker
    from .hyde import Hyde, make_default_hyde
except ImportError:
    from indexer import Indexer
    from embedder import Embedder
    from reranker import Reranker
    from hyde import Hyde, make_default_hyde


logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    id: str
    text: str
    score: float
    source: str = ""
    project: str = ""
    ts: str = ""
    confidence: float = 0.0
    entities: List[str] = field(default_factory=list)
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    last_access: str = ""
    access_count: int = 0
    # 阶段 2 调试字段
    decay_factor: float = 1.0
    rrf_score: float = 0.0
    # 阶段 2 rerank
    rerank_score: Optional[float] = None

    def to_dict(self) -> Dict:
        d = {
            "id": self.id,
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
            "project": self.project,
            "ts": self.ts,
            "confidence": self.confidence,
            "entities": self.entities,
            "rrf_score": round(self.rrf_score, 4),
            "decay_factor": round(self.decay_factor, 3),
            "access_count": self.access_count,
        }
        if self.rerank_score is not None:
            d["rerank_score"] = round(self.rerank_score, 4)
        return d


def _parse_ts(ts_str: str) -> Optional[datetime]:
    """容忍空 / 错误格式"""
    if not ts_str:
        return None
    try:
        # 支持 'YYYY-MM-DDTHH:MM:SS' 和 'YYYY-MM-DD'
        if "T" in ts_str:
            return datetime.fromisoformat(ts_str)
        return datetime.fromisoformat(ts_str + "T00:00:00")
    except (ValueError, TypeError):
        return None


class Retriever:
    """三路混合检索 + RRF 融合 + 可选 Rerank + 可选 HyDE"""

    def __init__(
        self,
        indexer: Indexer,
        embedder: Embedder,
        reranker: Reranker = None,
        hyde: Optional[Hyde] = None,
        rrf_k: int = 30,                  # 阶段 2 优化: 60→30，区分度更明显
        decay_tau_days: float = 90.0,     # 时间衰减常数（公式定义在 seed_memory.md）
        record_access: bool = True,      # 检索时自动递增 access_count
    ):
        self.indexer = indexer
        self.embedder = embedder
        self.reranker = reranker
        self.hyde = hyde  # None = 不用 HyDE
        self.rrf_k = rrf_k
        self.decay_tau_days = decay_tau_days
        self.record_access = record_access

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        source: Optional[str] = None,
        enable_decay: bool = True,
        candidates: int = 20,
        rerank: bool = False,
        rerank_top_n: int = 20,
        use_hyde: bool = False,
    ) -> List[RetrievalResult]:
        """
        1. 可选 HyDE Query 改写（use_hyde=True）
        2. 向量检索 Top-N（用 hyde_doc 或 query）
        3. BM25 检索 Top-N（始终用原 query，BM25 对短 query 更稳）
        4. RRF 融合
        5. 时间衰减重排（可选）
        6. 可选 Rerank（rerank=True 时启用）
        7. 返回 Top-K
        8. 自动记录 access_count（可选）
        """
        # 1. HyDE Query 改写
        search_text = query  # 默认用原 query
        if use_hyde and self.hyde is not None:
            try:
                search_text = self.hyde.generate(query)
                logger.debug("HyDE 改写: %d → %d chars", len(query), len(search_text))
            except (RuntimeError, ValueError) as e:
                logger.warning("HyDE 失败，回退到原 query: %s", e)
                search_text = query

        # 2. 向量检索（用 search_text，可能经 HyDE 改写）
        query_vec = self.embedder.embed(search_text).tolist()
        vec_hits = self.indexer.vector_search(query_vec, k=candidates, project=project)
        vec_by_id = {h["id"]: (i, h) for i, h in enumerate(vec_hits)}

        # 3. BM25（始终用原 query；BM25 对短 query 鲁棒）
        bm25_hits = self.indexer.bm25_search(query, k=candidates, project=project)
        bm25_by_id = {h["id"]: (i, h) for i, h in enumerate(bm25_hits)}

        # 3. 合并候选
        all_ids = set(vec_by_id.keys()) | set(bm25_by_id.keys())
        now = datetime.now()

        results: List[RetrievalResult] = []
        for cid in all_ids:
            vec_rank, vec_hit = vec_by_id.get(cid, (None, None))
            bm_rank, bm_hit = bm25_by_id.get(cid, (None, None))
            hit = vec_hit or bm_hit
            if not hit:
                continue

            # 元数据过滤
            if source and hit.get("source") != source:
                continue

            # entities 字段可能是 JSON 字符串
            ents = hit.get("entities", "[]")
            if isinstance(ents, str):
                try:
                    ents = json.loads(ents)
                except (json.JSONDecodeError, TypeError):
                    ents = []

            # RRF score
            rrf_score = 0.0
            if vec_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + vec_rank + 1)
            if bm_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + bm_rank + 1)

            # 时间衰减
            ts = hit.get("ts", "")
            ts_dt = _parse_ts(ts)
            access_count = int(hit.get("access_count", 0))
            decay = 1.0
            if enable_decay and ts_dt:
                delta_days = (now - ts_dt).total_seconds() / 86400.0
                decay = math.exp(-delta_days / self.decay_tau_days)
                # 访问频率加成（log 缩放防热门记忆霸榜）
                decay *= math.log1p(access_count + 1)

            # 最终分数
            confidence = float(hit.get("confidence", 0.0))
            # confidence 权重 0.3，避免高置信但内容不相关的记忆靠 conf 漂上去
            score = rrf_score * decay * (0.7 + 0.3 * confidence)

            results.append(RetrievalResult(
                id=cid,
                text=hit.get("text", ""),
                score=score,
                source=hit.get("source", ""),
                project=hit.get("project", ""),
                ts=ts,
                confidence=confidence,
                entities=ents,
                vector_rank=vec_rank,
                bm25_rank=bm_rank,
                last_access=hit.get("last_access", ""),
                access_count=access_count,
                decay_factor=decay,
                rrf_score=rrf_score,
            ))

        # 按 score 降序
        results.sort(key=lambda r: r.score, reverse=True)

        # 5. 可选 Rerank（用 RRF 池的前 N 重新打分）
        rerank_scores = {}
        if rerank and self.reranker is not None:
            rerank_input = results[:rerank_top_n]
            rerank_docs = [r.text for r in rerank_input]
            try:
                rs = self.reranker.score(query, rerank_docs)
                for r, s in zip(rerank_input, rs):
                    rerank_scores[r.id] = float(s)
                # 重排：rerank score 优先，原始 score 作 tie-break
                results.sort(
                    key=lambda r: (rerank_scores.get(r.id, 0.0), r.score),
                    reverse=True,
                )
            except (RuntimeError, OSError, ValueError) as e:
                # 静默降级：不 rerank
                logger.warning("Retriever.search rerank 失败，回退: %s", e)

        top_results = results[:top_k]
        # 回填 rerank_score
        for r in top_results:
            if r.id in rerank_scores:
                r.rerank_score = rerank_scores[r.id]
                # 用 rerank score 覆盖最终 score（rerank 才是真相关性）
                r.score = r.rerank_score

        # 6. 记录访问（仅记录 Top-K）
        if self.record_access and top_results:
            try:
                self.indexer.record_access([r.id for r in top_results])
                # 回填 +1（让 UI 显示一致）
                for r in top_results:
                    r.access_count += 1
            except sqlite3.Error as e:
                # 静默降级，检索不应被记录失败阻塞
                logger.debug("record_access 失败（已忽略）: %s", e)

        return top_results