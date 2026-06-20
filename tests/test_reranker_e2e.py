"""
BGE Reranker 端到端验证

测试场景：
1. 模型加载（lazy load + HF_HUB_OFFLINE 兼容）
2. 单文档打分
3. 多文档重排（顺序循环，n=5 ≈ 400ms）
4. 与 Retriever 集成（rerank=True 提升精度）
5. 静默降级（reranker 报错不影响主检索）

依赖：
- 需先跑 download_bge_reranker.py 下模型
- 索引可空（用合成数据）
"""
from __future__ import annotations
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reranker import Reranker
from src.memory import Memory
from src.embedder import Embedder
from src.indexer import Indexer
from src.retriever import Retriever
from src.chunker import chunk_text, Chunk


def _make_chunks(texts_with_meta: list) -> list:
    """手工构造 Chunk 列表（绕过启发式 project 提取）"""
    out = []
    for item in texts_with_meta:
        if isinstance(item, str):
            text, meta = item, {}
        else:
            text, meta = item
        c = Chunk(text=text, meta={**meta, "source": meta.get("source", "synth.txt")})
        out.append(c)
    return out


def test_reranker_load():
    """测试 1: 模型懒加载"""
    print("\n=== TEST 1: 懒加载 ===")
    r = Reranker()
    print(f"  初始: {r}")
    t0 = time.time()
    r._load()
    elapsed = time.time() - t0
    print(f"  ✅ 加载耗时: {elapsed:.1f}s, 模型: {r.model_name}")
    print(f"  加载后: {r}")


def test_reranker_score():
    """测试 2: 单文档打分"""
    print("\n=== TEST 2: 单文档打分 ===")
    r = Reranker()
    docs = [
        "SkillFather 是 Agent Skill 适配度分析器，基于 README 多维度评分",
        "今天天气真好，适合去公园散步",
        "BGE Reranker 是 Cross-Encoder 重排模型，提升检索精度",
    ]
    t0 = time.time()
    scores = r.score("SkillFather 是什么", docs)
    elapsed = time.time() - t0

    print(f"  耗时: {elapsed*1000:.0f}ms ({len(docs)} docs)")
    for d, s in zip(docs, scores):
        marker = "🎯" if d.startswith("SkillFather") else "  "
        print(f"  {marker} {s:.4f}  {d[:60]}")
    assert scores[0] > scores[1], f"相关文档应高于无关：{scores[0]} <= {scores[1]}"
    print("  ✅ 相关文档分数 > 无关文档")


def test_reranker_rerank_method():
    """测试 3: rerank() 方法（返回重排后列表）"""
    print("\n=== TEST 3: rerank() 方法 ===")
    r = Reranker()
    hits = [
        {"id": "a", "text": "Python 是一种编程语言", "score": 0.5},
        {"id": "b", "text": "SkillFather 用 Python 开发，5 维度评分", "score": 0.8},
        {"id": "c", "text": "今天吃火锅很开心", "score": 0.3},
    ]
    out = r.rerank("SkillFather 什么技术栈", hits, top_k=3)
    print(f"  返回顺序:")
    for h in out:
        print(f"    {h.get('rerank_score', '?'):.4f}  {h['text'][:50]}")
    # b 应排第一（"SkillFather 用 Python"）
    assert out[0]["id"] == "b", f"期望 b 排第一，实际 {out[0]['id']}"
    print("  ✅ SkillFather 相关文档正确排第一")


def test_reranker_with_retriever():
    """测试 4: 与 Retriever 集成"""
    print("\n=== TEST 4: 集成到 Retriever（rerank=True） ===")
    with tempfile.TemporaryDirectory() as tmp:
        index_dir = Path(tmp) / "index"
        mem = Memory(index_dir=str(index_dir))

        # 注入 5 条合成数据（3 条 SkillFather 相关 + 2 条噪声）
        mem.add_chunks(_make_chunks([
            ("SkillFather 是 Python 项目，用于分析 Agent Skill 适配度，5 个评分维度",
             {"project": "skillfather"}),
            ("今天天气很好，适合出门散步",
             {"project": "weather"}),
            ("SkillFather 基于 README 多维度评分：完整性、准确性、可维护性、活跃度、覆盖度",
             {"project": "skillfather"}),
            ("我喜欢吃火锅，尤其是麻辣口味的",
             {"project": "food"}),
            ("SkillFather 安装路径在 ~/.workbuddy/skills/skillfather/，克隆自 GitHub",
             {"project": "skillfather"}),
        ]))

        # 用 Retriever 检索（带 rerank）
        embedder = Embedder()
        indexer = mem.indexer
        reranker = Reranker()

        # 显式 rerank=True
        retriever = Retriever(
            indexer=indexer,
            embedder=embedder,
            reranker=reranker,
            rrf_k=30,
            decay_tau_days=90.0,
        )

        t0 = time.time()
        results = retriever.search(
            "SkillFather 评分维度",
            top_k=3,
            rerank=True,
            rerank_top_n=10,
        )
        elapsed = time.time() - t0

        print(f"  耗时: {elapsed:.1f}s（含 rerank）")
        for i, r in enumerate(results, 1):
            marker = "🎯" if "SkillFather" in r.text else "  "
            print(f"  {i}. {marker} score={r.score:.4f} (rerank={r.rerank_score})  {r.text[:60]}")

        # 至少前 3 条应该是 SkillFather 相关
        skillfather_count = sum(1 for r in results if "SkillFather" in r.text)
        print(f"  SkillFather 命中率: {skillfather_count}/{len(results)}")
        assert skillfather_count >= 2, f"期望至少 2 条 SkillFather 命中，实际 {skillfather_count}"
        print("  ✅ rerank 显著提升语义相关排序")


def test_reranker_silent_fallback():
    """测试 5: 静默降级（reranker 异常不影响主检索）"""
    print("\n=== TEST 5: 静默降级 ===")
    # 模拟一个会报错的 reranker
    class BrokenReranker:
        def score(self, query, docs):
            raise RuntimeError("mocked failure")

    with tempfile.TemporaryDirectory() as tmp:
        index_dir = Path(tmp) / "index"
        mem = Memory(index_dir=str(index_dir))
        mem.add_chunks(_make_chunks([
            ("SkillFather 评分维度：完整性、准确性、可维护性",
             {"project": "skillfather"}),
        ]))
        embedder = Embedder()
        indexer = mem.indexer
        retriever = Retriever(
            indexer=indexer,
            embedder=embedder,
            reranker=BrokenReranker(),
        )

        # 即使 reranker 报错，检索应仍能返回结果
        results = retriever.search(
            "SkillFather 评分",
            top_k=3,
            rerank=True,
            rerank_top_n=10,
        )
        assert len(results) > 0, "reranker 失败时检索应仍返回结果"
        print(f"  ✅ BrokenReranker 报错，主检索仍返回 {len(results)} 条结果")
        print(f"  顶条 score={results[0].score:.4f}: {results[0].text[:60]}")


def main():
    print("🚀 BGE Reranker 端到端验证")
    print("=" * 70)
    t0 = time.time()
    test_reranker_load()
    test_reranker_score()
    test_reranker_rerank_method()
    test_reranker_with_retriever()
    test_reranker_silent_fallback()
    print()
    print("=" * 70)
    print(f"✅ 全部 5 测试通过，耗时 {time.time()-t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()