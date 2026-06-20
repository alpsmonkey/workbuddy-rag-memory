"""
测试 HyDE Query 改写
- Mock LLM：返回固定模板
- 缓存命中
- Retriever.search(use_hyde=True) 集成
"""
from __future__ import annotations
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_hyde_mock_generate():
    """Mock HyDE 应返回模板填充文本"""
    from src.hyde import Hyde, MOCK_TEMPLATES

    h = Hyde(llm=None)
    result = h.generate("去重阈值")
    assert isinstance(result, str)
    assert "去重阈值" in result
    assert len(result) > len("去重阈值")
    # 应包含模板前缀
    assert any(t.split("「")[0] in result for t in MOCK_TEMPLATES)
    print(f"✅ Mock HyDE 生成: '{result}'")


def test_hyde_custom_llm():
    """自定义 LLM：必须被调用"""
    from src.hyde import Hyde

    called = []
    def my_llm(q):
        called.append(q)
        return f"假设文档：{q} 是用于..."

    h = Hyde(llm=my_llm)
    result = h.generate("SkillFather")
    assert "假设文档" in result
    assert "SkillFather" in result
    assert called == ["SkillFather"]
    print(f"✅ 自定义 LLM 正常调用")


def test_hyde_cache():
    """同 query 应命中缓存"""
    from src.hyde import Hyde

    call_count = [0]
    def slow_llm(q):
        call_count[0] += 1
        time.sleep(0.01)  # 模拟延迟
        return f"doc {q}"

    h = Hyde(llm=slow_llm, cache_ttl=60)

    t0 = time.time()
    r1 = h.generate("cached query")
    t1 = time.time()

    t2 = time.time()
    r2 = h.generate("cached query")
    t3 = time.time()

    assert r1 == r2
    assert call_count[0] == 1  # LLM 只被调一次
    # 缓存命中应该比第一次快
    assert (t3 - t2) < (t1 - t0) * 0.5
    print(f"✅ 缓存命中: {call_count[0]} 次 LLM 调用，2 次 generate")


def test_hyde_llm_failure_fallback():
    """LLM 抛错时降级到 mock"""
    from src.hyde import Hyde

    def broken_llm(q):
        raise RuntimeError("LLM server down")

    h = Hyde(llm=broken_llm)
    result = h.generate("test query")
    # 应该 fallback 到 mock，仍返回字符串
    assert isinstance(result, str)
    assert "test query" in result
    print(f"✅ LLM 失败降级: '{result}'")


def test_hyde_empty_query():
    """空 query 直接返回"""
    from src.hyde import Hyde

    h = Hyde()
    assert h.generate("") == ""
    assert h.generate("   ") == "   "
    print("✅ 空 query 直通")


def test_retriever_with_hyde():
    """Retriever.search(use_hyde=True) 集成"""
    from src.hyde import Hyde
    from src.memory import Memory

    class MockEmbedder:
        dim = 1024  # 与 LanceDB schema 一致
        backend = "mock"
        def embed(self, text):
            import numpy as np
            np.random.seed(hash(text) % 100)
            return np.random.rand(self.dim)

    with tempfile.TemporaryDirectory() as tmp:
        mem = Memory(index_dir=tmp + "/.idx", embedder=MockEmbedder())

        # 入库 3 条
        for t in [
            "## SkillFather\nSkillFather 是 Python 项目。",
            "## 去重策略\n余弦相似度阈值 0.92 触发合并。",
            "## 时间衰减\nscore = rrf × exp(-Δt/τ)。",
        ]:
            mem.add(t)

        # 创建 Hyde（带 mock LLM）
        hyde = Hyde(llm=None)  # mock
        mem.retriever.hyde = hyde

        # 不用 HyDE
        r1 = mem.search("SkillFather", top_k=3, use_hyde=False)
        # 用 HyDE
        r2 = mem.search("SkillFather", top_k=3, use_hyde=True)

        # 至少一种方式能召回（mock embedder 质量低，可能 0 命中也属正常）
        # 这里只验证调用链不报错
        assert isinstance(r1, list)
        assert isinstance(r2, list)
        print(f"✅ Retriever 集成: 无 HyDE={len(r1)} 条, 有 HyDE={len(r2)} 条")


def test_make_default_hyde():
    """工厂函数返回 Hyde 实例"""
    from src.hyde import Hyde, make_default_hyde

    h = make_default_hyde()
    assert isinstance(h, Hyde)
    assert h.llm is None
    print("✅ make_default_hyde()")


def main():
    tests = [
        test_hyde_mock_generate,
        test_hyde_custom_llm,
        test_hyde_cache,
        test_hyde_llm_failure_fallback,
        test_hyde_empty_query,
        test_retriever_with_hyde,
        test_make_default_hyde,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"  hyde: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())