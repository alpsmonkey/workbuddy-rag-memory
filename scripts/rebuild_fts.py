"""
重建 FTS5 索引（trigram tokenizer）
- 解决 CJK 整段被当成一个 token 的问题
- 删旧表 → 用新 schema 建表 → 从 chunks 表回填

用法:
    python scripts/rebuild_fts.py --index-dir <DIR>
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexer import Indexer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    args = parser.parse_args()

    idx = Indexer(args.index_dir)
    sqlite_path = idx.index_dir / "meta.db"
    print(f"📦 SQLite: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        # 1. 统计
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"📊 chunks 总数: {total}")

        # 2. 删旧 FTS
        print("🗑️  DROP 旧 FTS5 (unicode61)...")
        conn.execute("DROP TABLE IF EXISTS chunks_fts")
        conn.commit()

        # 3. 用新 schema 建（trigram）
        print("🔨 CREATE 新 FTS5 (trigram)...")
        conn.execute("""
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                id UNINDEXED,
                text,
                project,
                entities,
                tokenize='trigram'
            )
        """)
        conn.commit()

        # 4. 回填
        print("📥 回填 FTS5...")
        rows = conn.execute(
            "SELECT id, text, project, entities FROM chunks"
        ).fetchall()
        inserted = 0
        for r in rows:
            try:
                conn.execute(
                    "INSERT INTO chunks_fts(id, text, project, entities) VALUES (?, ?, ?, ?)",
                    (r[0], r[1], r[2], r[3])
                )
                inserted += 1
            except Exception as e:
                print(f"  ⚠️  {r[0]}: {e}")
        conn.commit()
        print(f"✓ 回填: {inserted}/{total}")

        # 5. 验证
        print("\n=== 验证 ===")
        for q in ['"路径约定"', '"记忆架构"', '"默认工作空间根路径"', '路径', '记忆']:
            try:
                rows = conn.execute(
                    "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 3",
                    (q,)
                ).fetchall()
                ids = [r[0][:30] for r in rows]
                print(f'  FTS {q} → {len(rows)} hits: {ids}')
            except Exception as e:
                print(f'  FTS {q} → ERROR: {e}')
    finally:
        conn.close()
    print("\n✓ 完成")


if __name__ == "__main__":
    main()