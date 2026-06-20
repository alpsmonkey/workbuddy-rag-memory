"""
写入拦截器：去重 + 合并决策
- 写入前算 embedding
- 查最近 N 条最相似
- 余弦相似度 > threshold 触发合并/跳过

返回决策: 'insert' / 'merge' / 'skip'

索引大小策略（阶段 1.5 强化）：
- 默认 search_k=50
- 实际查询 = min(search_k, count, MAX_SEARCH_K)
- MAX_SEARCH_K 默认 200，防大索引 O(n) 退化
- 大索引（> 1000）时降低 search_k 避免冗余
"""
from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
import os

try:
    from .chunker import Chunk
    from .indexer import Indexer
    from .embedder import Embedder
    from .config import (
        get_max_search_k_compat, get_large_index_threshold_compat,
        get_dedup_skip_low,
    )
except ImportError:
    from chunker import Chunk
    from indexer import Indexer
    from embedder import Embedder
    from config import (
        get_max_search_k_compat, get_large_index_threshold_compat,
        get_dedup_skip_low,
    )


logger = logging.getLogger(__name__)


class Decision(str, Enum):
    INSERT = "insert"   # 新增
    MERGE = "merge"     # 覆盖/合并到已存在
    SKIP = "skip"       # 跳过（重复）


# 硬上限：防大索引时 vector_search 变成 O(n)
# 优先 WB_RAG_MAX_SEARCH_K > DEDUP_MAX_SEARCH_K (legacy) > pyproject.toml > 200
MAX_SEARCH_K = get_max_search_k_compat()

# 大索引阈值：超过此值认为"已经很丰富"，减少 dedup 召回窗口
LARGE_INDEX_THRESHOLD = get_large_index_threshold_compat()

# skip 区间下限（低于此相似度直接 insert）
DEFAULT_SKIP_LOW = get_dedup_skip_low()


@dataclass
class DedupResult:
    decision: Decision
    existing_id: Optional[str] = None
    similarity: float = 0.0
    reason: str = ""


class Dedup:
    """去重拦截器"""

    def __init__(
        self,
        indexer: Indexer,
        embedder: Embedder,
        threshold: float = 0.92,
        search_k: int = 50,
    ):
        self.indexer = indexer
        self.embedder = embedder
        self.threshold = threshold
        self.search_k = search_k
        # 大索引模式下自动压缩 search_k，避免冗余扫描
        self.large_index_k = max(20, search_k // 2)

    def _effective_search_k(self) -> int:
        """
        根据实际索引大小动态调整 search_k
        - count=0: 0（无需查）
        - count<=search_k: count（用尽）
        - count<=MAX_SEARCH_K: search_k
        - 大索引（> LARGE_INDEX_THRESHOLD）: 用 large_index_k
        - 否则: MAX_SEARCH_K（硬上限）
        """
        count = self.indexer.count()
        if count == 0:
            return 0
        if count > LARGE_INDEX_THRESHOLD:
            return min(self.large_index_k, count, MAX_SEARCH_K)
        return min(self.search_k, count, MAX_SEARCH_K)

    def check(self, chunk: Chunk) -> DedupResult:
        """
        检查 chunk 是否重复

        规则：
        - 相似度 >= threshold: 决策 merge（高置信度覆盖低置信度）
        - 相似度 0.85~0.92: 决策 skip（语义相似但内容不同，宁可错杀）
        - 相似度 < 0.85: 决策 insert

        注意：hash 兜底模式下无语义信息，跳过去重直接 insert
        """
        count = self.indexer.count()
        if count == 0:
            return DedupResult(Decision.INSERT, reason="empty index")

        # hash 兜底模式：无语义，跳过去重
        if self.embedder.backend == "hash-fallback":
            return DedupResult(Decision.INSERT, reason="hash-fallback mode: no dedup")

        k = self._effective_search_k()
        if k == 0:
            return DedupResult(Decision.INSERT, reason="no effective search k")

        emb = self.embedder.embed(chunk.text)

        # 查最近 N 条（受索引大小动态限制）
        candidates = self.indexer.vector_search(emb.tolist(), k=k)

        if not candidates:
            return DedupResult(Decision.INSERT, reason="no candidates")

        # 取最相似一条
        best = candidates[0]
        best_id = best.get("id", "")
        best_vec = best.get("vector", [])
        best_conf = float(best.get("confidence", 0.5))

        sim = self._cosine(emb, np.array(best_vec, dtype=np.float32))

        # 标记大索引模式，便于诊断
        size_tag = f"[{count} chunks, k={k}{',large-mode' if count > LARGE_INDEX_THRESHOLD else ''}]"

        if sim >= self.threshold:
            # 高相似：合并
            if chunk.meta.get("confidence", 0.5) >= best_conf:
                return DedupResult(
                    Decision.MERGE,
                    existing_id=best_id,
                    similarity=sim,
                    reason=f"sim={sim:.3f} >= {self.threshold}, new conf higher {size_tag}"
                )
            else:
                return DedupResult(
                    Decision.SKIP,
                    existing_id=best_id,
                    similarity=sim,
                    reason=f"sim={sim:.3f} >= {self.threshold}, existing conf higher {size_tag}"
                )

        if sim >= 0.85:
            # 中等相似：跳过（防语义撞车）
            return DedupResult(
                Decision.SKIP,
                existing_id=best_id,
                similarity=sim,
                reason=f"sim={sim:.3f} in [0.85, {self.threshold}), semantically similar {size_tag}"
            )

        return DedupResult(
            Decision.INSERT,
            existing_id=best_id,
            similarity=sim,
            reason=f"sim={sim:.3f} < 0.85, new fact {size_tag}"
        )

    def write(self, chunk: Chunk) -> DedupResult:
        """检查并执行写入"""
        result = self.check(chunk)

        if result.decision == Decision.INSERT:
            emb = self.embedder.embed(chunk.text)
            ok = self.indexer.insert(chunk, emb.tolist())
            if not ok:
                result.reason += " | insert failed"

        elif result.decision == Decision.MERGE:
            emb = self.embedder.embed(chunk.text)
            ok = self.indexer.update(result.existing_id, chunk, emb.tolist())
            if not ok:
                result.reason += " | update failed"

        return result

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度（已归一化向量 = 点积）"""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
