"""
测试 fastapi server
- TestClient 模拟 HTTP 请求
- 验证 /health, /search, /add, /batch_search, /stats 端点
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# fastapi TestClient
try:
    from fastapi.testclient import TestClient
except ImportError:
    print("⚠️  fastapi.testclient 不在，需要 httpx")
    from httpx import Client as TestClient  # type: ignore


def make_test_app():
    """构造一个测试用的 app + Memory"""
    from server import create_app
    from src.memory import Memory

    class MockEmbedder:
        dim = 1024
        backend = "mock"
        def embed(self, text):
            import numpy as np
            np.random.seed(hash(text) % 100)
            return np.random.rand(self.dim)

    tmp = tempfile.mkdtemp()
    mem = Memory(index_dir=tmp, embedder=MockEmbedder())
    mem.add("SkillFather 是 Python 项目，用于分析 Agent Skill。")
    mem.add("去重阈值 0.92 触发合并决策。")
    mem.add("时间衰减公式 score = rrf × exp(-Δt/τ)。")
    app = create_app(mem)
    return app, tmp


def test_health():
    app, tmp = make_test_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.2.3"
    assert data["backend"] == "mock"
    print(f"✅ /health: {data}")


def test_stats():
    app, tmp = make_test_app()
    client = TestClient(app)
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_chunks" in data
    assert "backend" in data
    assert data["embedding_dim"] == 1024
    print(f"✅ /stats: total={data['total_chunks']} backend={data['backend']}")


def test_search():
    app, tmp = make_test_app()
    client = TestClient(app)
    r = client.post("/search", json={"query": "SkillFather", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "SkillFather"
    assert "hits" in data
    assert "duration_ms" in data
    assert isinstance(data["hits"], list)
    print(f"✅ /search: query='SkillFather' → {data['count']} hits, {data['duration_ms']:.1f}ms")


def test_add():
    app, tmp = make_test_app()
    client = TestClient(app)
    r = client.post("/add", json={"text": "测试新记忆", "source": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["decision"] in ("insert", "merge", "skip")
    assert "reason" in data
    print(f"✅ /add: decision={data['decision']} reason={data['reason']}")


def test_batch_search():
    app, tmp = make_test_app()
    client = TestClient(app)
    r = client.post("/batch_search", json={
        "queries": ["SkillFather", "去重", "时间衰减"],
        "top_k": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    assert all("hits" in r for r in data["results"])
    print(f"✅ /batch_search: 3 queries → {[r['count'] for r in data['results']]}")


def test_search_validation():
    """top_k 范围验证"""
    app, tmp = make_test_app()
    client = TestClient(app)
    r = client.post("/search", json={"query": "test", "top_k": 999})
    # 应该 422（pydantic 校验）
    assert r.status_code == 422
    print("✅ /search 参数校验（top_k > 50 → 422）")


def test_search_empty_query():
    """空 query 必须被拒"""
    app, tmp = make_test_app()
    client = TestClient(app)
    r = client.post("/search", json={"query": "", "top_k": 3})
    assert r.status_code == 422
    print("✅ /search 空 query → 422")


def main():
    tests = [
        test_health,
        test_stats,
        test_search,
        test_add,
        test_batch_search,
        test_search_validation,
        test_search_empty_query,
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
    print(f"  server: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())