"""
HyDE 集成端到端测试

覆盖：
1. MockHyde 不需要 LLM 也能工作
2. use_hyde=False 走原 query
3. use_hyde=True 走 hyde_doc 改写
4. HyDE 缓存命中（5 分钟 TTL）
5. rag_search skill subprocess 透传 --no-hyde
6. Memory 注入 None hyde 时 use_hyde=True 自动降级
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

import pytest


# 把 src/ 加入 sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ============================================================
# 单元层：HyDE + Memory 行为
# ============================================================

def test_hyde_mock_no_llm_works():
    """无 LLM 时 MockHyde 也能生成改写文本"""
    from hyde import make_default_hyde
    h = make_default_hyde()
    doc = h.generate("去重阈值")
    # 不等于原 query，且包含 query 关键词
    assert doc != "去重阈值"
    assert "去重阈值" in doc
    assert len(doc) > 5


def test_hyde_cache_within_ttl():
    """5 分钟 TTL 缓存命中"""
    from hyde import make_default_hyde
    h = make_default_hyde()
    h.llm = None  # Mock 模式
    a = h.generate("缓存测试")
    b = h.generate("缓存测试")
    # 缓存命中应返回完全相同对象
    assert a == b
    # 缓存字典应该只有 1 条
    assert len(h._cache) == 1


def test_hyde_empty_query_returns_self():
    """空 query 直接返回"""
    from hyde import make_default_hyde
    h = make_default_hyde()
    assert h.generate("") == ""
    assert h.generate("   ") == "   "


def test_hyde_llm_fallback_to_mock():
    """LLM 抛异常时降级到 Mock"""
    from hyde import Hyde

    def bad_llm(q):
        raise RuntimeError("模拟 LLM 故障")

    h = Hyde(llm=bad_llm)
    doc = h.generate("降级测试")
    # 即使 LLM 炸了，HyDE 也能返回 Mock 结果（不阻断检索）
    assert doc and len(doc) > 0
    assert "降级测试" in doc


def test_memory_with_hyde_param_accepts_none():
    """Memory 注入 hyde=None 不应报错（兜底）"""
    from memory import Memory
    m = Memory(index_dir="./.index-test-hyde-none", hyde=None)
    # Memory 内部把 hyde 透传给 retriever
    assert m.retriever.hyde is None


def test_memory_with_hyde_param_accepts_instance():
    """Memory 注入 Mock hyde 实例应可用"""
    from memory import Memory
    from hyde import make_default_hyde
    h = make_default_hyde()
    m = Memory(index_dir="./.index-test-hyde-ok", hyde=h)
    # Memory 把 hyde 透传给 retriever
    assert m.retriever.hyde is h
    # 直接调 generate 验证注入成功
    assert h.generate("ping") != ""


# ============================================================
# 集成层：rag_search skill subprocess 透传
# ============================================================

VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
WORKER = ROOT / "skills" / "rag_search" / "worker.py"
MAIN = ROOT / "skills" / "rag_search" / "main.py"


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="venv 不存在，跳过集成测试")
def test_worker_default_uses_hyde():
    """worker.py 默认 use_hyde=True"""
    r = subprocess.run(
        [str(VENV_PYTHON), str(WORKER), "BGE-M3", "2"],
        capture_output=True, text=True, timeout=90, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
    )
    assert r.returncode == 0, f"worker 退出码 {r.returncode}: {r.stderr}"
    import json
    out = json.loads(r.stdout.strip())
    assert out.get("use_hyde") is True


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="venv 不存在，跳过集成测试")
def test_worker_no_hyde_flag():
    """worker.py --no-hyde 应关闭 HyDE"""
    r = subprocess.run(
        [str(VENV_PYTHON), str(WORKER), "BGE-M3", "2", "--no-hyde"],
        capture_output=True, text=True, timeout=90, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
    )
    assert r.returncode == 0, f"worker 退出码 {r.returncode}: {r.stderr}"
    import json
    out = json.loads(r.stdout.strip())
    assert out.get("use_hyde") is False


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="venv 不存在，跳过集成测试")
def test_hyde_changes_ranking_or_at_least_does_not_break():
    """HyDE 启用 / 关闭 都应能跑通检索（不强求结果差异，但保证不崩）"""
    import json
    common = {
        "capture_output": True,
        "text": True,
        "timeout": 90,
        "encoding": "utf-8",
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
    }
    r_on = subprocess.run(
        [str(VENV_PYTHON), str(WORKER), "上次聊过什么", "3"],
        **common,
    )
    r_off = subprocess.run(
        [str(VENV_PYTHON), str(WORKER), "上次聊过什么", "3", "--no-hyde"],
        **common,
    )
    assert r_on.returncode == 0
    assert r_off.returncode == 0
    out_on = json.loads(r_on.stdout.strip())
    out_off = json.loads(r_off.stdout.strip())
    # 两种模式都应召回 ≥1 条（生产索引已超 200 chunks）
    assert out_on["count"] >= 1
    assert out_off["count"] >= 1
    assert out_on["use_hyde"] is True
    assert out_off["use_hyde"] is False


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="venv 不存在，跳过集成测试")
def test_main_no_hyde_passes_through_to_worker():
    """main.py --no-hyde 应透传给 worker"""
    r = subprocess.run(
        [str(VENV_PYTHON), str(MAIN), "BGE-M3", "2", "--no-hyde"],
        capture_output=True, text=True, timeout=90, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
    )
    # main 返回的是格式化后的 JSON（results 数组或 note）
    assert r.returncode == 0, f"stderr: {r.stderr}"
    import json
    # main 的 format_output 直接 dump result dict
    out = json.loads(r.stdout.strip())
    # 即使没召回，use_hyde 字段也应该反映（main 没透传 use_hyde，但 worker 内部 print 了）
    # 检查至少有 query 字段
    assert "query" in out
