"""
WorkBuddy 启动时跑的 oneshot ingest 脚本。

特点：
- 静默模式：不打印进度，只写日志（不影响 WorkBuddy 启动速度）
- 失败不抛异常：任何错误都返回 0，不影响 WorkBuddy 启动
- 增量模式：默认开启，已入库文件 0 chunks 处理
- 自动 cwd 修复：从 venv python 调用时显式指定 cwd=项目根

用法:
  python scripts/ingest_wb_memory_oneshot.py            # 跑一次（静默）
  python scripts/ingest_wb_memory_oneshot.py --verbose  # 显示进度
  python scripts/ingest_wb_memory_oneshot.py --dry-run  # 只看不写
"""
from __future__ import annotations
import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


# 让 import 找得到 src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = Path.home() / ".workbuddy" / "rag-bootstrap.log"


def _log(msg: str, verbose: bool = False) -> None:
    """日志：永远写文件，verbose 时同时打印"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except (OSError, ValueError, KeyError):
        pass  # 日志写不进去也不能影响主流程
    if verbose:
        print(line, file=sys.stderr)


def _resolve_workspace_root(start: Path) -> Path:
    """向上找 .workbuddy 目录，作为 workspace 根"""
    cur = start.resolve()
    for _ in range(10):  # 最多向上 10 层
        if (cur / ".workbuddy").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # 找不到就用 PROJECT_ROOT 自身（保持兼容）
    return PROJECT_ROOT


def run_once(verbose: bool = False, dry_run: bool = False) -> int:
    """跑一次 ingest，返回 0（成功）或 1（失败但不影响 WorkBuddy 启动）"""
    try:
        _log(f"=== 启动 ingest_wb_memory_oneshot (dry_run={dry_run}) ===", verbose)

        # 找 workspace 根（含 .workbuddy 目录的最近祖先）
        workspace_root = _resolve_workspace_root(PROJECT_ROOT)
        os.chdir(workspace_root)

        from src.memory import Memory, DEFAULT_WB_MEMORY_DIRS

        index_dir = Path.home() / ".workbuddy" / "rag-index"
        index_dir.mkdir(parents=True, exist_ok=True)

        # 设置离线环境（避免启动时连 HuggingFace）
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        mem = Memory(index_dir=str(index_dir))

        # 走 Memory 的 scan_workbuddy_memory 方法（增量模式 + 真实目录）
        t0 = datetime.now()
        result = mem.scan_workbuddy_memory(
            dirs=DEFAULT_WB_MEMORY_DIRS,
            ignore=None,
            project="bootstrap",
            dry_run=dry_run,
        )
        duration = (datetime.now() - t0).total_seconds()

        msg = (
            f"扫描完成: scanned={result.get('scanned', 0)} "
            f"skipped={result.get('skipped', 0)} "
            f"chunks_total={result.get('chunks_total', 0)} "
            f"insert={result.get('inserted', 0)} "
            f"merge={result.get('merged', 0)} "
            f"skip_dup={result.get('skipped_dup', 0)} "
            f"unchanged={result.get('unchanged', 0)} "
            f"duration={duration:.2f}s"
        )
        _log(msg, verbose)

        # 健康度快照（可选，避免启动太慢）
        try:
            health = mem.indexer.all_chunks()
            _log(f"索引现状: total={len(health)} chunks", verbose)
        except (OSError, RuntimeError, ValueError) as e:
            _log(f"健康度快照失败: {e}", verbose)

        _log("=== ingest_wb_memory_oneshot 完成 ===", verbose)
        print(msg)  # stdout 给 skill bridge 解析
        return 0

    except (OSError, RuntimeError, ValueError, KeyError, AttributeError) as e:
        # 任何失败都吞掉，写日志即可
        tb = traceback.format_exc()
        _log(f"❌ ingest 失败: {e}\n{tb}", verbose)
        return 1  # 非零但 WorkBuddy 启动脚本可以忽略


def main():
    parser = argparse.ArgumentParser(description="WorkBuddy 启动时的 RAG ingest（静默模式）")
    parser.add_argument("--verbose", action="store_true", help="输出到 stderr")
    parser.add_argument("--dry-run", action="store_true", help="只看不写索引")
    args = parser.parse_args()

    sys.exit(run_once(verbose=args.verbose, dry_run=args.dry_run))


if __name__ == "__main__":
    main()