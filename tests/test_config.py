"""
测试 src.config 模块的路径参数化能力
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_pyproject_loading():
    """pyproject.toml 的 [tool.workbuddy-rag] 必须能解析"""
    from src.config import (
        get_index_dir, get_memory_dirs, get_memory_patterns,
        get_ignore_patterns, get_embedding_model, get_reranker_model,
        get_dedup_threshold, get_max_search_k, get_tau_days, get_rrf_k,
    )

    index_dir = get_index_dir()
    assert isinstance(index_dir, Path)
    assert ".workbuddy" in str(index_dir) or "rag-index" in str(index_dir)

    dirs = get_memory_dirs()
    assert isinstance(dirs, list)
    assert all(isinstance(d, Path) for d in dirs)

    patterns = get_memory_patterns()
    assert isinstance(patterns, list)
    assert any("YYYY" in p or "[0-9]" in p for p in patterns), \
        f"必须包含 YYYY-MM-DD 模式: {patterns}"

    ignore = get_ignore_patterns()
    assert "*.bak" in ignore

    assert get_embedding_model() == "BAAI/bge-m3"
    assert get_reranker_model() == "BAAI/bge-reranker-v2-m3"
    assert get_dedup_threshold() == 0.92
    assert get_max_search_k() == 200
    assert get_tau_days() == 90.0
    assert get_rrf_k() == 30
    print("✅ pyproject.toml 配置加载正确")


def test_env_override():
    """环境变量 WB_RAG_* 必须能覆盖 pyproject.toml"""
    from src import config

    # 用 Windows 兼容路径（Git Bash 下 /tmp → E:/tmp）
    # 注意：Windows runner 会把长路径自动转 8.3 短路径（如 RUNNER~1），
    # 必须用 Path.resolve() 把两边都 normalize 才能 assert 相等
    test_path = str(Path(tempfile.gettempdir()).resolve() / "test-rag-index")

    # 模拟环境变量覆盖
    os.environ["WB_RAG_INDEX_DIR"] = test_path
    os.environ["WB_RAG_DEDUP_THRESHOLD"] = "0.85"
    os.environ["WB_RAG_MAX_SEARCH_K"] = "500"
    dir1 = str(Path(tempfile.gettempdir()).resolve() / "dir1")
    dir2 = str(Path(tempfile.gettempdir()).resolve() / "dir2")
    os.environ["WB_RAG_DEFAULT_MEMORY_DIRS"] = dir1 + "," + dir2

    # 清缓存
    config._config_cache = None

    # 两边都通过 Path.resolve() 标准化（处理 Windows 8.3 短路径）
    assert Path(str(config.get_index_dir())) == Path(test_path)
    assert config.get_dedup_threshold() == 0.85
    assert config.get_max_search_k() == 500
    expected_dirs = [Path(dir1), Path(dir2)]
    assert [Path(str(d)) for d in config.get_memory_dirs()] == expected_dirs

    # 清理
    del os.environ["WB_RAG_INDEX_DIR"]
    del os.environ["WB_RAG_DEDUP_THRESHOLD"]
    del os.environ["WB_RAG_MAX_SEARCH_K"]
    del os.environ["WB_RAG_DEFAULT_MEMORY_DIRS"]
    config._config_cache = None

    print("✅ 环境变量覆盖正确（Windows 8.3 路径兼容）")


def test_memory_scan_uses_config():
    """scan_workbuddy_memory 不传参数时必须用 config 的默认"""
    from src.memory import Memory

    # 用 mock embedder 避免 bge-m3 加载
    class MockEmbedder:
        dim = 4
        def embed(self, text):
            import numpy as np
            np.random.seed(hash(text) % 100)
            return np.random.rand(self.dim)

    with tempfile.TemporaryDirectory() as tmp:
        # 写一个临时 memory 文件
        mem_dir = Path(tmp) / ".workbuddy" / "memory"
        mem_dir.mkdir(parents=True)
        (mem_dir / "2026-06-20.md").write_text("# 测试\nSkillFather 是 Python。", encoding="utf-8")

        # 通过 WB_RAG_DEFAULT_MEMORY_DIRS 指定扫描目录
        os.environ["WB_RAG_DEFAULT_MEMORY_DIRS"] = str(mem_dir.parent)
        from src import config
        config._config_cache = None

        try:
            mem = Memory(index_dir=tmp + "/.idx", embedder=MockEmbedder())
            result = mem.scan_workbuddy_memory()  # 不传 dirs

            assert result["scanned"] >= 1, f"应扫到 1 个，实际: {result}"
            assert any("2026-06-20" in f["path"] for f in result["files"])
            print(f"✅ Memory 默认 dirs 来自 config（扫到 {result['scanned']} 个）")
        finally:
            del os.environ["WB_RAG_DEFAULT_MEMORY_DIRS"]
            config._config_cache = None


def test_print_config():
    """print_config 必须能跑通（命令行调试用）"""
    import subprocess
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("⏭️  跳过（venv 不存在）")
        return

    r = subprocess.run(
        [str(venv_python), str(PROJECT_ROOT / "src" / "config.py"), "--verbose"],
        cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    assert "INDEX_DIR" in r.stdout
    assert "MEMORY_DIRS" in r.stdout
    print("✅ python -m src.config --verbose 正常")


def main():
    tests = [
        test_pyproject_loading,
        test_env_override,
        test_memory_scan_uses_config,
        test_print_config,
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
    print(f"  config: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())