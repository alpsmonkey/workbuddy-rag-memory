"""
dedup 测试 — 重点验证 2026-06-20 新增的 source 层级守门：
- summary-over-full 必须 SKIP（防退化）
- full-over-summary 必须 MERGE（允许升级）
- 同层级仍按 confidence 比较
"""
import sys
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dedup import Dedup, Decision, SOURCE_LEVELS, _level_of
from src.chunker import Chunk
from src.embedder import Embedder


class FakeEmbedder:
    """不依赖真实模型：返回固定维度的单位向量（让 cosine=1.0 强制触发 merge/skip 分支）"""
    backend = "sentence-transformers"  # 跳过 hash-fallback 兜底
    dim = 4

    def embed(self, text: str):
        # 关键：返回相同非零向量，cosine=1.0 必命中 sim >= threshold 分支
        # 不能用全零向量，否则会被 _cosine 的除零兜底返回 0.0
        return np.ones(self.dim, dtype=np.float32)


class FakeIndexer:
    """最小化 Indexer 替身：只暴露 vector_search / count / insert / update"""
    def __init__(self, chunks: list = None):
        self._chunks = chunks or []
        self._inserted = []
        self._updated = []

    def count(self) -> int:
        return len(self._chunks)

    def vector_search(self, vector, k: int = 50, project: str = None) -> list:
        # 全部返回，dedup 会取 candidates[0]
        return self._chunks[:k]

    def insert(self, chunk: Chunk, vector) -> bool:
        self._inserted.append(chunk.id)
        return True

    def update(self, chunk_id: str, chunk: Chunk, vector) -> bool:
        self._updated.append(chunk_id)
        return True


def make_chunk(text: str, source: str, confidence: float = 0.8) -> Chunk:
    """构造带 source / confidence 元数据的 chunk"""
    return Chunk(
        text=text,
        meta={"source": source, "confidence": confidence},
    )


# ==================== 层级映射单元测试 ====================

def test_source_levels_mapping():
    """source 标签 → 层级映射必须按设计文档"""
    assert _level_of("user-memory") == "summary"
    assert _level_of("workspace-log") == "full"
    assert _level_of("conversation") == "full"
    assert _level_of("unknown") == "full"
    assert _level_of("") == "full"  # 空字符串默认 full（保守）
    assert _level_of("nonexistent") == "full"  # 未知默认 full


# ==================== 边界场景 1: summary-over-full 必须 SKIP ====================

def test_summary_over_full_blocked():
    """
    场景：MEMORY.md（summary 级）的精简条目试图覆盖工作日志（full 级）的展开版本
    期望：SKIP（防退化，保护工作日志的细节）
    """
    # 已存在：full 级（workspace-log 工作日志）
    existing = {
        "id": "existing_full_001",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.7,
        "source": "workspace-log",
    }
    indexer = FakeIndexer(chunks=[existing])
    dedup = Dedup(indexer=indexer, embedder=FakeEmbedder(), threshold=0.92)

    # 新来：summary 级（user-memory MEMORY.md 精简条目）
    new_chunk = make_chunk(
        text="SkillFather 用 Python",  # 文本本身不重要，cosine=1.0
        source="user-memory",
        confidence=0.95,  # 即使 confidence 更高也被层级规则拦截
    )

    result = dedup.check(new_chunk)

    assert result.decision == Decision.SKIP, f"summary-over-full 必须 SKIP，实际 {result.decision}"
    assert "summary-over-full blocked" in result.reason
    assert "user-memory" in result.reason and "workspace-log" in result.reason


# ==================== 边界场景 2: full-over-summary 必须 MERGE ====================

def test_full_over_summary_merges():
    """
    场景：工作日志（full 级）的展开版本覆盖 MEMORY.md（summary 级）的精简条目
    期望：MERGE（升级：骨架被血肉补全）
    """
    # 已存在：summary 级（user-memory MEMORY.md）
    existing = {
        "id": "existing_summary_001",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.6,
        "source": "user-memory",
    }
    indexer = FakeIndexer(chunks=[existing])
    dedup = Dedup(indexer=indexer, embedder=FakeEmbedder(), threshold=0.92)

    # 新来：full 级（workspace-log 工作日志）
    new_chunk = make_chunk(
        text="SkillFather 用 Python，5 维评分...",
        source="workspace-log",
        confidence=0.7,
    )

    result = dedup.check(new_chunk)

    assert result.decision == Decision.MERGE, f"full-over-summary 必须 MERGE，实际 {result.decision}"
    assert "upgrade summary->full" in result.reason


# ==================== 边界场景 3: 同层级仍按 confidence 比较 ====================

def test_same_level_higher_conf_merges():
    """full 级新 chunk 比已有 full 级 confidence 高 → MERGE"""
    existing = {
        "id": "existing_full",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.5,
        "source": "workspace-log",
    }
    indexer = FakeIndexer(chunks=[existing])
    dedup = Dedup(indexer=indexer, embedder=FakeEmbedder(), threshold=0.92)

    new_chunk = make_chunk("...", source="workspace-log", confidence=0.8)
    result = dedup.check(new_chunk)

    assert result.decision == Decision.MERGE


def test_same_level_lower_conf_skips():
    """full 级新 chunk confidence 低于已有 → SKIP"""
    existing = {
        "id": "existing_full",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.9,
        "source": "workspace-log",
    }
    indexer = FakeIndexer(chunks=[existing])
    dedup = Dedup(indexer=indexer, embedder=FakeEmbedder(), threshold=0.92)

    new_chunk = make_chunk("...", source="workspace-log", confidence=0.5)
    result = dedup.check(new_chunk)

    assert result.decision == Decision.SKIP


def test_summary_over_summary_higher_conf_merges():
    """summary 级互相比较也按 confidence"""
    existing = {
        "id": "existing_summary",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.5,
        "source": "user-memory",
    }
    indexer = FakeIndexer(chunks=[existing])
    dedup = Dedup(indexer=indexer, embedder=FakeEmbedder(), threshold=0.92)

    new_chunk = make_chunk("...", source="user-memory", confidence=0.8)
    result = dedup.check(new_chunk)

    assert result.decision == Decision.MERGE


# ==================== 反向：full-over-full + 同 conf = SKIP ====================

def test_full_over_summary_lower_conf_skips():
    """full-over-summary 即使升级，新 conf 低于旧 conf 仍 SKIP"""
    existing = {
        "id": "existing_summary",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.9,  # 旧 summary 置信度高
        "source": "user-memory",
    }
    indexer = FakeIndexer(chunks=[existing])
    dedup = Dedup(indexer=indexer, embedder=FakeEmbedder(), threshold=0.92)

    new_chunk = make_chunk("...", source="workspace-log", confidence=0.5)
    result = dedup.check(new_chunk)

    assert result.decision == Decision.SKIP
    assert "existing conf higher" in result.reason


# ==================== 实战场景：MEMORY.md 与工作日志双向保护 ====================

def test_pengge_real_world_workflow():
    """
    实战模拟鹏哥的 RAG 工作流：
    1. 工作日志写入（full 级）→ MERGE 到 MEMORY.md 骨架（summary）
    2. MEMORY.md 后又精简 → SKIP（不覆盖已有血肉）
    """
    indexer = FakeIndexer()
    dedup = Dedup(indexer=indexer, embedder=FakeEmbedder(), threshold=0.92)

    # Step 1: 已有 MEMORY.md 骨架
    indexer._chunks.append({
        "id": "mem_rag_basic",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.6,
        "source": "user-memory",
    })

    # Step 2: 工作日志写入展开版本
    log_chunk = make_chunk(
        "RAG 用 bge-m3 + reranker + lancedb",
        source="workspace-log",
        confidence=0.75,
    )
    r1 = dedup.check(log_chunk)
    assert r1.decision == Decision.MERGE, "工作日志升级 MEMORY.md 应被允许"
    assert "upgrade" in r1.reason

    # Step 3: 模拟 ingest 阶段把工作日志写进索引后，索引更新（best 变成 full）
    indexer._chunks[0] = {
        "id": "mem_rag_basic",
        "vector": np.ones(4, dtype=np.float32).tolist(),
        "confidence": 0.75,
        "source": "workspace-log",  # 现在已升级为 full 级
    }

    # Step 4: MEMORY.md 又被精简（conf 更高但仍是 summary）
    mem_update = make_chunk(
        "RAG = bge-m3 + reranker",
        source="user-memory",
        confidence=0.95,
    )
    r2 = dedup.check(mem_update)
    assert r2.decision == Decision.SKIP, "MEMORY.md 精简条目不能反向覆盖已升级的工作日志细节"
    assert "summary-over-full blocked" in r2.reason
