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

source 层级边界（2026-06-20 强化 — 防 D 方案 / rag_search 撞车）：
- 记忆按信息密度分两级：
    SUMMARY 级：user-memory        (MEMORY.md 长期事实，骨架级，D 方案预注入用)
    FULL    级：workspace-log / conversation / unknown (血肉级，rag_search 按需深挖用)
- 合并时层级守门（防止细节丢失或反向退化）：
    新 full     + 旧 summary → MERGE  (允许升级：骨架被血肉补全)
    新 summary  + 旧 full    → SKIP   (防退化：保留已有细节，不让骨架反向覆盖血肉)
    同层级                 → 按 confidence 比较（原逻辑）
- 效果：
    MEMORY.md 的精简条目不会被工作日志/对话的展开版本无脑覆盖（升级安全）
    工作日志/对话的精细版本也不会被 MEMORY.md 的简短更新覆盖（反向安全）
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


# source 层级映射（2026-06-20 强化）
# 依据 chunker.py:117-128 detect_source() 的返回标签
# 'user-memory' (MEMORY.md) 是骨架级；其他都是血肉级
SOURCE_LEVELS: Dict[str, str] = {
    "user-memory":   "summary",   # 骨架级（D 方案预注入用）
    "workspace-log": "full",      # 血肉级（rag_search 按需深挖）
    "conversation":  "full",      # 血肉级
    "unknown":       "full",      # 未知默认按 full 对待（保守策略：宁保细节）
}


def _level_of(source: str) -> str:
    """查 source 的层级（summary / full），未知默认 full（保细节）"""
    return SOURCE_LEVELS.get(source or "unknown", "full")


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
        - 相似度 >= threshold:
            * summary-over-full → SKIP（防退化，2026-06-20 新增）
            * 同层级 / full-over-summary → 按 confidence 比较：
                - 新 conf >= 旧 conf → MERGE
                - 否则 → SKIP
        - 相似度 0.85~0.92: SKIP（语义相似但内容不同，宁可错杀）
        - 相似度 < 0.85: INSERT

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
        best_source = best.get("source", "unknown")

        # 新 chunk 的元数据
        new_source = chunk.meta.get("source", "unknown")
        new_conf = float(chunk.meta.get("confidence", 0.5))
        new_level = _level_of(new_source)
        best_level = _level_of(best_source)

        sim = self._cosine(emb, np.array(best_vec, dtype=np.float32))

        # 标记大索引模式，便于诊断
        size_tag = f"[{count} chunks, k={k}{',large-mode' if count > LARGE_INDEX_THRESHOLD else ''}]"

        if sim >= self.threshold:
            # === 层级守门（2026-06-20 新增）===
            # 防 summary 覆盖 full → 保留细节（不让 MEMORY.md 的精简条目覆盖工作日志）
            if new_level == "summary" and best_level == "full":
                return DedupResult(
                    Decision.SKIP,
                    existing_id=best_id,
                    similarity=sim,
                    reason=(
                        f"summary-over-full blocked: "
                        f"new({new_source}) vs existing({best_source}) {size_tag}"
                    ),
                )
            # 同层级 / full-over-summary（升级）→ 按 confidence 比较
            if new_conf >= best_conf:
                level_tag = "" if new_level == best_level else f", upgrade {best_level}->{new_level}"
                return DedupResult(
                    Decision.MERGE,
                    existing_id=best_id,
                    similarity=sim,
                    reason=f"sim={sim:.3f} >= {self.threshold}, new conf higher{level_tag} {size_tag}",
                )
            else:
                return DedupResult(
                    Decision.SKIP,
                    existing_id=best_id,
                    similarity=sim,
                    reason=f"sim={sim:.3f} >= {self.threshold}, existing conf higher {size_tag}",
                )

        if sim >= 0.85:
            # 中等相似：跳过（防语义撞车）
            return DedupResult(
                Decision.SKIP,
                existing_id=best_id,
                similarity=sim,
                reason=f"sim={sim:.3f} in [0.85, {self.threshold}), semantically similar {size_tag}",
            )

        return DedupResult(
            Decision.INSERT,
            existing_id=best_id,
            similarity=sim,
            reason=f"sim={sim:.3f} < 0.85, new fact {size_tag}",
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
