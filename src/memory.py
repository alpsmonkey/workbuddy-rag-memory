"""
统一入口: 装配 Embedder / Indexer / Dedup / Retriever
"""
from __future__ import annotations
import os
import re
import json
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from .embedder import Embedder, get_default_embedder
    from .indexer import Indexer
    from .dedup import Dedup, Decision, DedupResult
    from .retriever import Retriever, RetrievalResult
    from .chunker import Chunk, chunk_text
    from .reranker import Reranker
    from .hyde import Hyde, make_default_hyde
    from .config import get_index_dir as _get_index_dir, get_dedup_threshold as _get_dedup_threshold
except ImportError:
    from embedder import Embedder, get_default_embedder
    from indexer import Indexer
    from dedup import Dedup, Decision, DedupResult
    from retriever import Retriever, RetrievalResult
    from chunker import Chunk, chunk_text
    from reranker import Reranker
    from hyde import Hyde, make_default_hyde
    from config import get_index_dir as _get_index_dir, get_dedup_threshold as _get_dedup_threshold


# 兼容旧代码：保留原常量（指向 .config 默认值）
# 推荐用 src.config 模块代替
try:
    from .config import (
        get_index_dir as _get_index_dir,
        get_memory_dirs as _get_memory_dirs,
        get_memory_patterns as _get_memory_patterns,
        get_ignore_patterns as _get_ignore_patterns,
        get_dedup_threshold as _get_dedup_threshold,
    )
    # 提供惰性求值的常量（保持旧代码 import 不报错）
    DEFAULT_WB_MEMORY_DIRS = ["~/.workbuddy/memory", "./.workbuddy/memory"]
    DEFAULT_WB_MEMORY_PATTERNS = [
        "*_memory.md",
        "MEMORY.md",
        "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*.md",
    ]
    DEFAULT_IGNORE_PATTERNS = [
        "*.bak", "*.tmp.*", "*~", "*.swp", ".DS_Store",
        "README.md", "CHANGELOG.md", "SKILL.md", "INSTALL.md",
        "TROUBLESHOOTING.md", "LICENSE.md", "CONTRIBUTING.md",
    ]
except ImportError:
    # standalone 跑时（无包结构）
    DEFAULT_WB_MEMORY_DIRS = ["~/.workbuddy/memory", "./.workbuddy/memory"]
    DEFAULT_WB_MEMORY_PATTERNS = [
        "*_memory.md",
        "MEMORY.md",
        "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*.md",
    ]
    DEFAULT_IGNORE_PATTERNS = [
        "*.bak", "*.tmp.*", "*~", "*.swp", ".DS_Store",
    ]

# 增量扫描状态文件名
INGEST_STATE_FILENAME = ".ingest_state.json"


class Memory:
    """高层 API：开箱即用"""

    def __init__(
        self,
        index_dir: str = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        hyde: Optional[Hyde] = None,
        dedup_threshold: float = None,
    ):
        self.embedder = embedder or get_default_embedder()
        # 优先级：参数 > WB_RAG_INDEX_DIR > INDEX_DIR (legacy) > pyproject.toml > ./.index
        if index_dir:
            self.index_dir = Path(index_dir)
        else:
            try:
                self.index_dir = _get_index_dir()
            except (OSError, KeyError, AttributeError):
                self.index_dir = Path(os.getenv("INDEX_DIR", "./.index"))
        self.indexer = Indexer(str(self.index_dir))

        # 阈值优先级：参数 > WB_RAG_DEDUP_THRESHOLD > DEDUP_THRESHOLD (legacy) > pyproject.toml > 0.92
        if dedup_threshold is not None:
            effective_threshold = dedup_threshold
        else:
            legacy = os.getenv("DEDUP_THRESHOLD")
            if legacy is not None:
                import warnings as _w
                _w.warn("DEDUP_THRESHOLD 已 deprecated，请用 WB_RAG_DEDUP_THRESHOLD", DeprecationWarning)
                effective_threshold = float(legacy)
            else:
                try:
                    effective_threshold = _get_dedup_threshold()
                except (OSError, KeyError, ValueError, AttributeError):
                    effective_threshold = 0.92
        self.dedup = Dedup(
            self.indexer,
            self.embedder,
            threshold=effective_threshold,
        )
        self.retriever = Retriever(
            self.indexer, self.embedder, reranker=reranker, hyde=hyde,
        )

        # 增量状态文件（按目录存，避免冲突）
        self._ingest_state_path = self.index_dir / INGEST_STATE_FILENAME

    def add(self, text: str, source: str = None) -> DedupResult:
        chunks = chunk_text(text, source=source)
        if not chunks:
            return DedupResult(Decision.SKIP, reason="no valid chunks after splitting")
        if len(chunks) == 1:
            return self.dedup.write(chunks[0])
        last = None
        for c in chunks:
            last = self.dedup.write(c)
        return last or DedupResult(Decision.INSERT, reason="multi-chunk batch")

    def add_file(self, file_path: str) -> int:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return 0
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gbk", errors="ignore")
        except (OSError, ValueError, KeyError):
            return 0
        chunks = chunk_text(text, source=str(path))
        if not chunks:
            return 0
        results = self.add_chunks(chunks)
        return sum(1 for r in results if r.decision in (Decision.INSERT, Decision.MERGE))

    def add_chunks(self, chunks: List[Chunk]) -> List[DedupResult]:
        return [self.dedup.write(c) for c in chunks]

    # ================================================================
    # 增量扫描状态管理
    # ================================================================

    def _load_ingest_state(self) -> dict:
        if not self._ingest_state_path.exists():
            return {}
        try:
            return json.loads(self._ingest_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            return {}

    def _save_ingest_state(self, state: dict) -> None:
        self._ingest_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._ingest_state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _file_signature(path: Path) -> dict:
        st = path.stat()
        return {"mtime": st.st_mtime, "size": st.st_size}

    @staticmethod
    def _is_changed(sig: dict, prev: dict) -> bool:
        if not prev:
            return True
        return prev.get("mtime") != sig["mtime"] or prev.get("size") != sig["size"]

    # ================================================================

    def scan_workbuddy_memory(
        self,
        dirs: Optional[List[str]] = None,
        ignore: Optional[List[str]] = None,
        project: Optional[str] = None,
        incremental: bool = True,
        dry_run: bool = False,
        patterns: Optional[List[str]] = None,
    ) -> dict:
        """
        扫描 WorkBuddy 真实记忆目录并入库

        Args:
            dirs: 目录列表
            ignore: 忽略的 glob 模式
            project: 强制 project 标签
            incremental: 增量模式（mtime+size 跳过未变文件，默认 True）
            dry_run: 只扫不入库

        Returns: dict（详见源码）
        """
        from collections import Counter

        # 优先用参数 > src.config (pyproject.toml) > 模块常量
        if dirs is None:
            try:
                dirs = [str(d) for d in _get_memory_dirs()]
            except (OSError, KeyError, AttributeError):
                dirs = DEFAULT_WB_MEMORY_DIRS
        if ignore is None:
            try:
                ignore = _get_ignore_patterns()
            except (OSError, KeyError, AttributeError):
                ignore = DEFAULT_IGNORE_PATTERNS
        if patterns is None:
            try:
                patterns = _get_memory_patterns()
            except (OSError, KeyError, AttributeError):
                patterns = DEFAULT_WB_MEMORY_PATTERNS

        resolved_dirs = []
        for raw in dirs:
            p = Path(raw).expanduser().resolve()
            if p.exists() and p.is_dir():
                resolved_dirs.append(p)

        if not resolved_dirs:
            return {
                "scanned": 0, "skipped": 0, "chunks_total": 0,
                "inserted": 0, "merged": 0, "skipped_dup": 0,
                "unchanged": 0, "files": [],
                "note": "no valid workbuddy memory dirs found",
            }

        all_files: List[Path] = []
        for d in resolved_dirs:
            for pattern in patterns:
                all_files.extend(d.rglob(pattern))
        all_files = sorted(set(all_files))

        skip_count = 0
        kept_files: List[Path] = []
        for f in all_files:
            name = f.name
            if any(re.match(p.replace("*", ".*").replace("?", "."), name) for p in ignore):
                skip_count += 1
                continue
            kept_files.append(f)

        # 加载增量状态
        state = self._load_ingest_state() if incremental else {}
        new_state = dict(state) if incremental else {}

        report = {
            "scanned": len(all_files),
            "skipped": skip_count,
            "chunks_total": 0,
            "inserted": 0,
            "merged": 0,
            "skipped_dup": 0,
            "unchanged": 0,
            "files": [],
            "incremental": incremental,
            "dry_run": dry_run,
        }

        for f in kept_files:
            sig = self._file_signature(f)
            key = str(f)
            is_changed = self._is_changed(sig, state.get(key, {}))

            if not is_changed:
                report["unchanged"] += 1
                new_state[key] = sig
                continue

            if dry_run:
                report["files"].append({
                    "path": str(f),
                    "changed": True,
                    "mtime": sig["mtime"],
                    "size": sig["size"],
                })
                new_state[key] = sig
                continue

            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = f.read_text(encoding="gbk", errors="ignore")
            except (OSError, PermissionError):
                text = ""
            chunks = chunk_text(text, source=str(f))
            if project:
                for c in chunks:
                    c.meta["project"] = project
            results = self.add_chunks(chunks) if chunks else []

            decisions = Counter(r.decision.value for r in results)
            report["chunks_total"] += len(results)
            report["inserted"] += decisions.get("insert", 0)
            report["merged"] += decisions.get("merge", 0)
            report["skipped_dup"] += decisions.get("skip", 0)
            report["files"].append({
                "path": str(f),
                "chunks": len(results),
                "decisions": dict(decisions),
                "mtime": sig["mtime"],
                "size": sig["size"],
            })
            new_state[key] = sig

        # 清理已不存在的文件
        if incremental and not dry_run:
            existing_keys = {str(p) for p in kept_files}
            for k in list(new_state.keys()):
                if k not in existing_keys:
                    del new_state[k]
            self._save_ingest_state(new_state)

        return report

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: Optional[str] = None,
        source: Optional[str] = None,
        enable_decay: bool = True,
        rerank: bool = False,
        candidates: int = 20,
        use_hyde: bool = False,
    ) -> List[RetrievalResult]:
        return self.retriever.search(
            query, top_k=top_k, project=project, source=source,
            enable_decay=enable_decay, rerank=rerank, candidates=candidates,
            use_hyde=use_hyde,
        )

    def stats(self) -> dict:
        return self.indexer.stats()

    def __repr__(self):
        return (
            f"Memory(index_dir={self.indexer.index_dir}, "
            f"total={self.indexer.count()}, embedder={self.embedder})"
        )