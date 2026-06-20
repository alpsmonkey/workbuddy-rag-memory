"""
dedup + indexer + retriever 集成测试
"""
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory
from src.embedder import Embedder
from src.indexer import Indexer
from src.dedup import Dedup, Decision
from src.chunker import Chunk
import numpy as np


def test_full_flow():
    """端到端：插入 → 检索 → 去重"""
    tmpdir = tempfile.mkdtemp(prefix="rag_test_")
    try:
        mem = Memory(index_dir=tmpdir)
        print(f"  使用后端: {mem.embedder.backend}")

        # 1. 插入 3 条
        r1 = mem.add("SkillFather 项目决定用 Python 做 5 维评分引擎，2026-05-28 立项")
        r2 = mem.add("WorkBuddy 记忆架构有 8 个真实缺点待修复")
        r3 = mem.add("RAG 增强记忆的优化分 3 个阶段实施")
        assert r1.decision == Decision.INSERT
        assert r2.decision == Decision.INSERT
        assert r3.decision == Decision.INSERT

        # 2. 检索
        results = mem.search("SkillFather 用什么语言", top_k=3)
        assert len(results) >= 1, "应能召回 SkillFather"
        print(f"  检索召回: {len(results)} 条")
        for r in results[:2]:
            print(f"    - {r.text[:60]}...")

        # 3. 去重：插入高度相似的句子
        r_dup = mem.add("SkillFather 项目决定用 Python 做 5 维评分引擎，2026-05-28 立项")
        print(f"  重复插入决策: {r_dup.decision.value} (sim={r_dup.similarity:.3f})")
        # hash fallback 模式下 dedup 禁用，决策为 insert（这是设计行为）
        # 真实语义后端下应为 MERGE 或 SKIP
        if mem.embedder.backend != "hash-fallback":
            assert r_dup.decision in (Decision.MERGE, Decision.SKIP), \
                f"语义后端下应触发去重，实际 {r_dup.decision}"

        # 4. 索引计数
        count = mem.indexer.count()
        assert count >= 3, f"应有至少 3 条，实际 {count}"

        # 5. 健康度
        stats = mem.stats()
        assert stats["total"] == count
        print(f"  统计: {stats}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_rrf_fusion():
    """验证 RRF 融合逻辑"""
    tmpdir = tempfile.mkdtemp(prefix="rag_rrf_")
    try:
        mem = Memory(index_dir=tmpdir)
        mem.add("RAG 用 RRF 融合向量和 BM25 的检索结果")
        mem.add("向量检索用 lancedb")
        mem.add("BM25 用 SQLite FTS5")

        # 模糊 query：只匹配 BM25（精确词）
        r1 = mem.search("SQLite FTS5", top_k=3)
        # 模糊 query：只匹配向量
        r2 = mem.search("融合算法", top_k=3)
        # 综合 query：两路都中
        r3 = mem.search("RRF 融合检索", top_k=3)

        print(f"  关键词检索: {len(r1)} 条")
        print(f"  语义检索:   {len(r2)} 条")
        print(f"  综合检索:   {len(r3)} 条")
        assert len(r3) >= 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("🧪 集成测试 1: 端到端流程")
    test_full_flow()
    print("\n🧪 集成测试 2: RRF 融合")
    test_rrf_fusion()
    print("\n✅ 集成测试全部通过")
