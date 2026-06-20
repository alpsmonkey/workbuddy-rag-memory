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


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """核心检索逻辑"""
    if not INDEX_DIR.exists():
        return []

    try:
        from memory import Memory
        memory = Memory(index_dir=str(INDEX_DIR), dedup_threshold=0.92)
        results = memory.search(query, top_k=top_k)
        return [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
    except Exception:
        return []


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    if not query:
        print(json.dumps({"query": "", "count": 0, "results": [], "note": "empty query"}))
        return
    top_k = max(1, min(20, top_k))
    results = search(query, top_k)
    out = {"query": query, "count": len(results), "results": results}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()