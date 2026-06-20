"""
路径配置加载器 - 推广级关键：从 pyproject.toml 读配置 + 环境变量覆盖

优先级（从高到低）：
1. 环境变量（WB_RAG_*）
2. pyproject.toml [tool.workbuddy-rag] 段
3. 内置默认值

用法:
  from src.config import get_config, get_index_dir, get_memory_dirs

  index_dir = get_index_dir()  # Path 对象，自动 expanduser
  dirs = get_memory_dirs()      # List[Path]
"""
from __future__ import annotations
import os
import sys
import tomllib
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"

_config_cache: Optional[dict] = None


def _load_pyproject() -> dict:
    """读 pyproject.toml 的 [tool.workbuddy-rag] 段"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if not PYPROJECT_PATH.exists():
        _config_cache = {}
        return _config_cache

    try:
        with open(PYPROJECT_PATH, "rb") as f:
            data = tomllib.load(f)
        _config_cache = data.get("tool", {}).get("workbuddy-rag", {})
    except Exception:
        _config_cache = {}
    return _config_cache


def _env_override(name: str, default):
    """读环境变量 WB_RAG_<NAME>，支持类型推断"""
    env_name = f"WB_RAG_{name.upper()}"
    val = os.getenv(env_name)
    if val is None:
        return default
    # 类型推断
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        try:
            return int(val)
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(val)
        except ValueError:
            return default
    if isinstance(default, list):
        # 逗号分隔
        return [x.strip() for x in val.split(",") if x.strip()]
    return val


def get_config(key: str, default=None):
    """读单个配置：环境变量 > pyproject.toml > default"""
    cfg = _load_pyproject()
    pyproject_val = cfg.get(key, default)
    return _env_override(key, pyproject_val)


def get_index_dir() -> Path:
    """共享索引目录（默认 ~/.workbuddy/rag-index）"""
    raw = get_config("index_dir", "~/.workbuddy/rag-index")
    return Path(raw).expanduser().resolve()


def get_memory_dirs() -> List[Path]:
    """默认扫描的真实记忆源（用户级 + 项目级）"""
    raw_list = get_config("default_memory_dirs", [
        "~/.workbuddy/memory",
        "./.workbuddy/memory",
    ])
    return [Path(d).expanduser() for d in raw_list]


def get_memory_patterns() -> List[str]:
    """默认扫描的 glob 模式"""
    return get_config("default_memory_patterns", [
        "*_memory.md",
        "MEMORY.md",
        "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*.md",
    ])


def get_ignore_patterns() -> List[str]:
    """默认忽略的文件名"""
    return get_config("default_ignore_patterns", [
        "*.bak", "*.tmp.*", "*~", "*.swp", ".DS_Store",
        "README.md", "CHANGELOG.md", "SKILL.md",
    ])


def get_embedding_model() -> str:
    return get_config("embedding_model", "BAAI/bge-m3")


def get_reranker_model() -> str:
    return get_config("reranker_model", "BAAI/bge-reranker-v2-m3")


def get_dedup_threshold() -> float:
    return float(get_config("dedup_threshold", 0.92))


def get_max_search_k() -> int:
    return int(get_config("max_search_k", 200))


def get_tau_days() -> float:
    return float(get_config("tau_days", 90.0))


def get_rrf_k() -> int:
    return int(get_config("rrf_k", 30))


def print_config(verbose: bool = False) -> None:
    """打印当前配置（调试用）"""
    print("=" * 60)
    print(f"WorkBuddy RAG 配置 (来源: pyproject.toml + 环境变量覆盖)")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  INDEX_DIR: {get_index_dir()}")
    print(f"  MEMORY_DIRS: {[str(d) for d in get_memory_dirs()]}")
    if verbose:
        print(f"  MEMORY_PATTERNS: {get_memory_patterns()}")
        print(f"  IGNORE_PATTERNS: {get_ignore_patterns()}")
        print(f"  EMBEDDING_MODEL: {get_embedding_model()}")
        print(f"  RERANKER_MODEL: {get_reranker_model()}")
        print(f"  DEDUP_THRESHOLD: {get_dedup_threshold()}")
        print(f"  MAX_SEARCH_K: {get_max_search_k()}")
        print(f"  TAU_DAYS: {get_tau_days()}")
        print(f"  RRF_K: {get_rrf_k()}")
    print("=" * 60)


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    print_config(verbose=verbose)