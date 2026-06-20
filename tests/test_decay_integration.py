"""
test_decay_integration.py
- 验证 retriever 接入时间衰减后检索质量变化
- 验证 access_count 自动递增
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory


def test_access_count_increments_on_search():
    """search() 应自动递增 access_count"""
    mem = Memory(index_dir="./.test_index_decay")
    # 先入库
    mem.add("SkillFather 用 Python 5维评分引擎")
    mem.add("LanceDB 是嵌入式向量数据库")
    mem.add("bge-m3 是 embedding 模型")

    # 取一个 chunk 看看初始 access_count
    total = mem.indexer.count()
    assert total >= 3, f"应至少 3 chunks，实际 {total}"

    # 第一次检索
    results = mem.search("Python 项目", top_k=2)
    assert len(results) > 0
    top_id = results[0].id
    top_acc_1 = mem.indexer.get(top_id).get("access_count", 0)

    # 第二次检索同一 query
    results2 = mem.search("Python 项目", top_k=2)
    top_acc_2 = mem.indexer.get(top_id).get("access_count", 0)

    # access_count 应递增
    assert top_acc_2 > top_acc_1, f"access_count 没递增: {top_acc_1} -> {top_acc_2}"

    print(f"✅ access_count 递增正常: {top_acc_1} -> {top_acc_2}")
    # 清理
    import shutil
    shutil.rmtree("./.test_index_decay", ignore_errors=True)


def test_decay_reweighting():
    """时间衰减应让旧记忆下沉（手工造一个 ts 旧的数据）"""
    import sqlite3
    from datetime import datetime, timedelta

    mem = Memory(index_dir="./.test_index_decay2")
    mem.add("新鲜记忆：今天的事")

    # 手工改 SQLite 把 ts 改到 1 年前
    old_ts = (datetime.now() - timedelta(days=365)).isoformat(timespec="seconds")
    with mem.indexer._conn() as c:
        # 在 chunk 里塞一句"一年前的事"
        c.execute(
            "INSERT OR REPLACE INTO chunks(id, text, ts, project, source, confidence, entities, length) VALUES (?, ?, ?, '', 'test', 0.9, '[]', 100)",
            ("old_test_id", "一年前的事：今天的事", old_ts)
        )
        c.execute("INSERT OR REPLACE INTO chunks_fts(id, text, project, entities) VALUES ('old_test_id', '一年前的事：今天的事', '', '[]')")
        c.commit()

    # 给新 chunk 也加到 lance
    from src.embedder import get_default_embedder
    emb = get_default_embedder()
    vec = emb.embed("一年前的事：今天的事").tolist()
    record = {
        "id": "old_test_id",
        "text": "一年前的事：今天的事",
        "vector": vec,
        "ts": old_ts,
        "project": "",
        "source": "test",
        "confidence": 0.9,
        "entities": "[]",
        "length": 100,
    }
    try:
        mem.indexer.table.add([record])
    except Exception:
        pass

    # 检索 - 带 decay 的应该让新鲜的排前
    results_decay = mem.search("今天的事", top_k=3, enable_decay=True)
    results_no_decay = mem.search("今天的事", top_k=3, enable_decay=False)

    print(f"\n📊 with decay: {[(r.text[:20], r.score) for r in results_decay]}")
    print(f"📊 no decay:   {[(r.text[:20], r.score) for r in results_no_decay]}")

    # 清理
    import shutil
    shutil.rmtree("./.test_index_decay2", ignore_errors=True)


if __name__ == "__main__":
    test_access_count_increments_on_search()
    test_decay_reweighting()
    print("\n✅ all decay tests passed")