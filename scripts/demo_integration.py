"""
WorkBuddy 同进程 RAG 集成 Demo

验证：
1. 同进程 import 是否可行（vs HTTP）
2. 触发条件：哪些 query 该触发 RAG（避免无脑吃 2.7 GB RAM）
3. 实测检索质量

用法：
  cd E:/workspace/2026-06-19-19-09-28/workbuddy-rag-memory
  PYTHONPATH=. <managed_python> scripts/demo_integration.py
"""
from __future__ import annotations
import os
import sys
import time
import re
from pathlib import Path

# 强制离线（嵌入/重排模型都在本地缓存）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ============================================================
# 1. 触发条件判断（避免每次 query 都吃 2.7 GB）
# ============================================================
HISTORY_TRIGGERS = [
    r"之前", r"上次", r"上次聊", r"前几天", r"昨天", r"前天",
    r"记得", r"还记得", r"你记不记得", r"提过", r"聊过", r"说过",
    r"那时候", r"那一次", r"当时", r"去年", r"上月",
    r"以前", r"过往", r"曾经", r"过去",
]
_TRIGGER_RE = re.compile("|".join(HISTORY_TRIGGERS))


def needs_history(query: str) -> bool:
    """轻量判断：是否需要检索历史记忆"""
    if _TRIGGER_RE.search(query):
        return True
    # 短 query + 第一人称也可能要历史
    if len(query) <= 12 and any(w in query for w in ["我", "咱", "你"]):
        return True
    return False


# ============================================================
# 2. Lazy Singleton：第一次才真正 import + 加载模型
# ============================================================
class RagMemory:
    """同进程 import 的 RAG 包装器

    设计要点：
    - lazy：第一次 search() 才加载 Embedder（约 2.7 GB RAM）
    - 单例：多次实例化不会重复加载
    - 异常隔离：底层异常上抛给 WorkBuddy 主进程 catch
    """
    _instance = None

    def __init__(self):
        from src.memory import Memory
        # 不传 index_dir，走 ~/.workbuddy/rag-index/ 默认
        self._mem = Memory()
        self._loaded_at = time.time()
        print(f"[RAG] lazy loaded in {time.time()-self._loaded_at:.2f}s")

    @classmethod
    def get(cls) -> "RagMemory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def search(self, query: str, top_k: int = 5, use_decay: bool = True):
        """检索历史记忆

        返回 List[dict]: [{text, score, source, project, access_count}, ...]
        """
        results = self._mem.search(
            query=query,
            top_k=top_k,
            candidates=20,
            enable_decay=use_decay,
            rerank=False,  # 默认不启用（省 2.3 GB RAM + 提速 5x）
        )
        return [
            {
                "text": r.text,
                "score": round(r.score, 4),
                "source": getattr(r, "source", "?"),
                "project": getattr(r, "project", "?"),
            }
            for r in results
        ]


# ============================================================
# 3. 演示
# ============================================================
def main():
    test_queries = [
        ("WorkBuddy 的安装步骤", False),    # 不应该触发 RAG
        ("上次聊过什么", True),             # 应该触发 RAG
        ("我之前问过 SAP 的问题吗", True),   # 应该触发 RAG
        ("BGE-M3 是怎么加载的", False),      # 不应该触发（技术细节，不查历史）
        ("昨天有没有聊到 hash fallback", True),  # 应该触发
    ]

    print("=" * 70)
    print("WorkBuddy RAG · 同进程集成 Demo")
    print("=" * 70)
    print()

    for query, expect_rag in test_queries:
        needs = needs_history(query)
        flag = "✓" if needs == expect_rag else "✗"
        print(f"  [{flag}] 触发判断: {'RAG' if needs else 'skip'} | {query!r}")

    print()
    print("=" * 70)
    print("实测检索（前 2 个查询命中历史）：")
    print("=" * 70)
    print()

    # 只对真正需要历史的 query 触发 RAG
    history_queries = [q for q, need in test_queries if needs_history(q)]

    if not history_queries:
        print("（所有 query 都没命中触发条件，跳过 RAG 加载）")
        return

    print(f"→ 第一次检索才会触发 lazy load（约 10s）...")
    t0 = time.time()
    rag = RagMemory.get()
    print(f"→ Memory 实例就绪，耗时 {time.time()-t0:.1f}s")
    print()

    for q in history_queries:
        print(f"Q: {q}")
        results = rag.search(q, top_k=3)
        if not results:
            print("  (no results)")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['project']}/{r['source']}]")
            print(f"     {r['text'][:120]}{'...' if len(r['text']) > 120 else ''}")
        print()


if __name__ == "__main__":
    main()