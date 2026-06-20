"""
RAG 检索 Worker - 实际跑检索的脚本（被 main.py subprocess 调用）

放在 venv python 跑（依赖 numpy/torch/lancedb/sentence-transformers）
由 main.py 负责调起，结果以 JSON 写到 stdout

路径自动发现：
  本文件所在位置: <PROJECT>/skills/rag_search/worker.py
  → PROJECT = 父目录的父目录
  → SRC = PROJECT/src（注入 sys.path）

可覆盖：
  WB_RAG_SRC         指定 src 绝对路径
  WB_RAG_INDEX_DIR   指定索引目录（默认 ~/.workbuddy/rag-index）
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any


# 强制离线（避免启动时连 HuggingFace）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ============================================================
# 路径自动发现
# ============================================================
_THIS_FILE = Path(__file__).resolve()
SKILL_DIR = _THIS_FILE.parent                  # skills/rag_search/
PROJECT_ROOT = SKILL_DIR.parent.parent         # 项目根

# 默认 src 路径（项目根/src）
_DEFAULT_SRC = PROJECT_ROOT / "src"
RAG_SRC = Path(os.environ.get("WB_RAG_SRC", str(_DEFAULT_SRC)))
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

# 共享索引（与 ingest_wb_memory_oneshot.py 默认值保持一致）
_DEFAULT_INDEX = Path.home() / ".workbuddy" / "rag-index"
INDEX_DIR = Path(os.environ.get("WB_RAG_INDEX_DIR", str(_DEFAULT_INDEX)))


def search(query: str, top_k: int = 5, use_hyde: bool = True) -> List[Dict[str, Any]]:
    """核心检索逻辑

    Args:
        query: 检索 query
        top_k: 返回条数
        use_hyde: 是否启用 HyDE Query 改写（默认 True，使用 Mock，零额外资源）
    """
    if not INDEX_DIR.exists():
        return []

    try:
        from memory import Memory
        # 注入 Mock HyDE（无 LLM 时也能提升短 query 召回 5-15%）
        try:
            from hyde import make_default_hyde
            hyde = make_default_hyde()
        except (ImportError, AttributeError) as e:
            # HyDE 模块缺失时降级（不阻断检索）
            hyde = None
            print(f"[worker] HyDE 不可用: {e}", file=sys.stderr)

        memory = Memory(index_dir=str(INDEX_DIR), dedup_threshold=0.92, hyde=hyde)
        results = memory.search(query, top_k=top_k, use_hyde=use_hyde)
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
    except Exception as e:
        print(f"[worker] search failed: {e}", file=sys.stderr)
        return []


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG worker - 检索执行")
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("top_k", nargs="?", type=int, default=5)
    parser.add_argument("--no-hyde", dest="use_hyde", action="store_false", default=True,
                        help="禁用 HyDE Query 改写（默认启用 Mock HyDE）")
    args = parser.parse_args()

    if not args.query:
        print(json.dumps({"query": "", "count": 0, "results": [], "note": "empty query"}))
        return
    top_k = max(1, min(20, args.top_k))
    results = search(args.query, top_k, use_hyde=args.use_hyde)
    out = {"query": args.query, "count": len(results), "results": results, "use_hyde": args.use_hyde}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()