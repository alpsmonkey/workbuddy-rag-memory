"""
三索引存储:
1. LanceDB - 向量检索
2. SQLite FTS5 - 关键词检索
3. SQLite - 元数据过滤

三者通过 chunk_id 关联
"""
from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

import lancedb
import pyarrow as pa
import re

try:
    from .chunker import Chunk
except ImportError:
    from chunker import Chunk


SCHEMA_VECTOR = pa.schema([
    ("id", pa.string()),
    ("text", pa.string()),
    ("vector", pa.list_(pa.float32(), 1024)),  # bge-m3 dim
    ("ts", pa.string()),
    ("project", pa.string()),
    ("source", pa.string()),
    ("confidence", pa.float32()),
    ("entities", pa.string()),  # JSON array as string
    ("length", pa.int32()),
])

SCHEMA_META = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    ts TEXT,
    project TEXT,
    source TEXT,
    confidence REAL,
    entities TEXT,
    length INTEGER,
    access_count INTEGER DEFAULT 0,
    last_access TEXT
);
"""

SCHEMA_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id UNINDEXED,
    text,
    project,
    entities,
    tokenize = 'trigram'
);
"""


class Indexer:
    """三索引联合存储"""

    def __init__(self, index_dir: str = "./.index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # LanceDB
        self.db = lancedb.connect(str(self.index_dir / "lance"))
        self.table_name = "chunks"
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=SCHEMA_VECTOR)
        else:
            self.table = self.db.open_table(self.table_name)

        # SQLite (元数据 + FTS)
        self.sqlite_path = self.index_dir / "meta.db"
        self._init_sqlite()

    def _init_sqlite(self):
        with self._conn() as c:
            c.executescript(SCHEMA_META)
            c.executescript(SCHEMA_FTS)
            c.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def insert(self, chunk: Chunk, vector: List[float]) -> bool:
        """插入单条 chunk（去重逻辑在 dedup 层处理）"""
        try:
            # 1. LanceDB
            record = {
                "id": chunk.id,
                "text": chunk.text,
                "vector": vector,
                "ts": chunk.meta.get("ts", ""),
                "project": chunk.meta.get("project") or "",
                "source": chunk.meta.get("source", ""),
                "confidence": float(chunk.meta.get("confidence", 0.5)),
                "entities": json.dumps(chunk.meta.get("entities", []), ensure_ascii=False),
                "length": int(chunk.meta.get("length", len(chunk.text))),
            }
            self.table.add([record])

            # 2. SQLite 元数据
            with self._conn() as c:
                c.execute(
                    """INSERT OR REPLACE INTO chunks
                       (id, text, ts, project, source, confidence, entities, length)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chunk.id, chunk.text, record["ts"], record["project"],
                     record["source"], record["confidence"],
                     record["entities"], record["length"])
                )
                # 3. FTS
                c.execute(
                    "INSERT INTO chunks_fts(id, text, project, entities) VALUES (?, ?, ?, ?)",
                    (chunk.id, chunk.text, record["project"], record["entities"])
                )
                c.commit()
            return True
        except Exception as e:
            print(f"[Indexer.insert] 失败: {e}")
            return False

    def update(self, chunk_id: str, chunk: Chunk, vector: List[float]) -> bool:
        """更新现有 chunk（用于去重合并）"""
        try:
            # LanceDB 删除后插入
            self.table.delete(f"id = '{chunk_id}'")
            record = {
                "id": chunk_id,
                "text": chunk.text,
                "vector": vector,
                "ts": chunk.meta.get("ts", ""),
                "project": chunk.meta.get("project") or "",
                "source": chunk.meta.get("source", ""),
                "confidence": float(chunk.meta.get("confidence", 0.5)),
                "entities": json.dumps(chunk.meta.get("entities", []), ensure_ascii=False),
                "length": int(chunk.meta.get("length", len(chunk.text))),
            }
            self.table.add([record])

            with self._conn() as c:
                c.execute(
                    """UPDATE chunks SET text=?, ts=?, project=?, source=?,
                       confidence=?, entities=?, length=? WHERE id=?""",
                    (chunk.text, record["ts"], record["project"], record["source"],
                     record["confidence"], record["entities"], record["length"], chunk_id)
                )
                c.execute("DELETE FROM chunks_fts WHERE id=?", (chunk_id,))
                c.execute(
                    "INSERT INTO chunks_fts(id, text, project, entities) VALUES (?, ?, ?, ?)",
                    (chunk_id, chunk.text, record["project"], record["entities"])
                )
                c.commit()
            return True
        except Exception as e:
            print(f"[Indexer.update] 失败: {e}")
            return False

    def vector_search(self, vector: List[float], k: int = 20,
                      project: str = None) -> List[Dict]:
        """LanceDB 向量检索"""
        try:
            q = self.table.search(vector).limit(k)
            if project:
                q = q.where(f"project = '{project}'")
            results = q.to_list()
            return results
        except Exception as e:
            print(f"[Indexer.vector_search] 失败: {e}")
            return []

    def bm25_search(self, query: str, k: int = 20, project: str = None) -> List[Dict]:
        """FTS5 关键词检索（CJK trigram 友好）
        - 拆 query 为 3-char 滑窗（覆盖 CJK）
        - 非 CJK 单词作为 phrase
        - OR 拼装
        """
        try:
            safe = re.sub(r"[^\w\s\u4e00-\u9fff\-]", " ", query)
            if not safe.strip():
                return []

            terms = []
            for token in safe.split():
                if not token:
                    continue
                cjk_chars = [c for c in token if "\u4e00" <= c <= "\u9fff"]
                if len(cjk_chars) >= 3:
                    # CJK: 生成所有 3-char 滑窗
                    for i in range(len(cjk_chars) - 2):
                        terms.append(f'"{("".join(cjk_chars[i:i+3]))}"')
                elif len(cjk_chars) >= 1:
                    # CJK 但太短（1-2字）：跳过（trigram 无法匹配）
                    pass
                else:
                    # 非 CJK
                    if len(token) >= 2:
                        terms.append(f'"{token}"')

            if not terms:
                return []
            fts_query = " OR ".join(terms)

            with self._conn() as c:
                if project:
                    rows = c.execute(
                        """SELECT c.id, c.text, c.project, c.source, c.confidence,
                                  c.entities, c.ts, c.length, bm25(chunks_fts) AS score
                           FROM chunks_fts f
                           JOIN chunks c ON c.id = f.id
                           WHERE chunks_fts MATCH ? AND c.project = ?
                           ORDER BY score LIMIT ?""",
                        (fts_query, project, k)
                    ).fetchall()
                else:
                    rows = c.execute(
                        """SELECT c.id, c.text, c.project, c.source, c.confidence,
                                  c.entities, c.ts, c.length, bm25(chunks_fts) AS score
                           FROM chunks_fts f
                           JOIN chunks c ON c.id = f.id
                           WHERE chunks_fts MATCH ?
                           ORDER BY score LIMIT ?""",
                        (fts_query, k)
                    ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[Indexer.bm25_search] 失败: {e}")
            return []

    def get(self, chunk_id: str) -> Optional[Dict]:
        """按 ID 获取"""
        with self._conn() as c:
            row = c.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
            if row:
                d = dict(row)
                d["entities"] = json.loads(d.get("entities", "[]"))
                return d
        return None

    def delete(self, chunk_id: str) -> bool:
        """按 ID 删除（三索引联动）"""
        try:
            # 1. LanceDB
            self.table.delete(f"id = '{chunk_id}'")
            # 2. SQLite 元数据 + FTS
            with self._conn() as c:
                c.execute("DELETE FROM chunks WHERE id=?", (chunk_id,))
                c.execute("DELETE FROM chunks_fts WHERE id=?", (chunk_id,))
                c.commit()
            return True
        except Exception as e:
            print(f"[Indexer.delete] 失败 {chunk_id}: {e}")
            return False

    def record_access(self, chunk_ids: List[str]) -> int:
        """
        记录访问：递增 access_count + 更新 last_access
        返回成功更新的行数
        - 用于时间衰减公式的 log(1+access_count) 项
        - 必须传 ID 列表，避免每次查全表
        """
        if not chunk_ids:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with self._conn() as c:
                placeholders = ",".join("?" * len(chunk_ids))
                c.execute(
                    f"""UPDATE chunks
                        SET access_count = access_count + 1,
                            last_access = ?
                        WHERE id IN ({placeholders})""",
                    [now] + list(chunk_ids)
                )
                c.commit()
                return c.total_changes
        except Exception as e:
            print(f"[Indexer.record_access] 失败: {e}")
            return 0

    def count(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) as n FROM chunks").fetchone()
            return row["n"]

    def stats(self) -> Dict[str, Any]:
        """健康度统计"""
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
            by_source = c.execute(
                "SELECT source, COUNT(*) as n FROM chunks GROUP BY source"
            ).fetchall()
            by_project = c.execute(
                "SELECT project, COUNT(*) as n FROM chunks WHERE project != '' GROUP BY project"
            ).fetchall()
        return {
            "total": total,
            "by_source": [dict(r) for r in by_source],
            "by_project": [dict(r) for r in by_project],
        }

    def all_chunks(self) -> List[Dict]:
        """取出所有 chunks（用于健康度分析、蒸馏）"""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, text, ts, project, source, confidence, entities, length, access_count, last_access FROM chunks"
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["entities"] = json.loads(d.get("entities", "[]"))
            except Exception:
                d["entities"] = []
            results.append(d)
        return results

    def by_project(self, project: str = None) -> Dict[str, List[Dict]]:
        """按 project 分组（project=None 时返回所有）"""
        chunks = self.all_chunks()
        grouped = {}
        for c in chunks:
            key = c.get("project") or "(未分类)"
            if project and key != project:
                continue
            grouped.setdefault(key, []).append(c)
        return grouped
