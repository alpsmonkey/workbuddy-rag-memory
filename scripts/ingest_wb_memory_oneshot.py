"""
WorkBuddy 启动时跑的 oneshot ingest 脚本。

特点：
- 静默模式：不打印进度，只写日志（不影响 WorkBuddy 启动速度）
- 失败不抛异常：任何错误都返回 0，不影响 WorkBuddy 启动
- 增量模式：默认开启，已入库文件 0 chunks 处理
- 自动 cwd 修复：从 venv python 调用时显式指定 cwd=项目根
- 蒸馏兜底（v0.2.4）：检查上次蒸馏时间，过期就自动跑
  解决"电脑没开机时 Task Scheduler 漏跑"的问题

用法:
  python scripts/ingest_wb_memory_oneshot.py            # 跑一次（静默）
  python scripts/ingest_wb_memory_oneshot.py --verbose  # 显示进度
  python scripts/ingest_wb_memory_oneshot.py --dry-run  # 只看不写
  python scripts/ingest_wb_memory_oneshot.py --skip-distill  # 跳过蒸馏检查
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path


# 让 import 找得到 src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_PATH = Path.home() / ".workbuddy" / "rag-bootstrap.log"

# 蒸馏兜底配置
DISTILL_STATE_FILENAME = ".distill_state.json"
DEFAULT_DISTILL_INTERVAL_HOURS = 24  # 超过 24h 就补跑
DEFAULT_DISTILL_MIN_CHUNKS = 50      # 索引 > 50 chunks 才值得蒸馏（避免冷启动浪费）


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


# ============================================================================
# 蒸馏兜底（解决电脑不开机时 03:00 蒸馏漏跑的问题）
# ============================================================================

def _distill_state_path(index_dir: Path) -> Path:
    """蒸馏状态文件路径（按 index_dir 分目录存）"""
    return index_dir / DISTILL_STATE_FILENAME


def _load_distill_state(index_dir: Path) -> dict:
    """读蒸馏状态：{last_distill_ts: float, last_total_chunks: int}"""
    state_file = _distill_state_path(index_dir)
    if not state_file.exists():
        return {"last_distill_ts": 0.0, "last_total_chunks": 0}
    try:
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, KeyError):
        return {"last_distill_ts": 0.0, "last_total_chunks": 0}


def _save_distill_state(index_dir: Path, total_chunks: int) -> None:
    """写蒸馏状态"""
    state_file = _distill_state_path(index_dir)
    try:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump({
                "last_distill_ts": time.time(),
                "last_total_chunks": total_chunks,
            }, f)
    except (OSError, ValueError, KeyError):
        pass  # 状态写不进去也不能影响主流程


def _should_distill(
    state: dict,
    interval_hours: float = DEFAULT_DISTILL_INTERVAL_HOURS,
    min_chunks: int = DEFAULT_DISTILL_MIN_CHUNKS,
    current_chunks: int = 0,
) -> tuple[bool, str]:
    """判断是否需要补跑蒸馏

    Returns:
        (should_run, reason)
    """
    last_ts = state.get("last_distill_ts", 0.0)

    # 从未蒸馏过 → 跑（首次运行）
    if last_ts == 0.0:
        return True, "从未蒸馏过（首次）"

    # 距上次蒸馏超过 interval_hours → 跑
    elapsed = time.time() - last_ts
    if elapsed > interval_hours * 3600:
        last_dt = datetime.fromtimestamp(last_ts).strftime("%Y-%m-%d %H:%M")
        return True, f"距上次 {last_dt} 已 {elapsed/3600:.1f}h > {interval_hours}h"

    # 索引太小不值得蒸馏（冷启动场景）
    if current_chunks < min_chunks:
        return False, f"索引仅 {current_chunks} chunks < {min_chunks} 阈值，跳过"

    return False, f"距上次蒸馏仅 {elapsed/3600:.1f}h，无需补跑"


def _run_distill(index_dir: Path, verbose: bool = False) -> dict:
    """实际跑一次 distill.py

    Returns:
        {"ok": bool, "purged": int, "groups": int, "duration_s": float}
    """
    distill_script = PROJECT_ROOT / "scripts" / "distill.py"
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

    if not venv_python.exists():
        return {"ok": False, "reason": "venv not found"}

    t0 = time.time()
    try:
        r = subprocess.run(
            [str(venv_python), str(distill_script), "--index-dir", str(index_dir), "--verbose"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟上限
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
        )
        duration = time.time() - t0

        # 解析蒸馏报告（distill.py 末尾会打印 "实际清理: N"）
        purged = 0
        groups = 0
        for line in (r.stdout or "").splitlines():
            if "实际清理:" in line:
                try:
                    purged = int(line.split("实际清理:")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
            if "总 chunks:" in line:
                try:
                    groups = int(line.split("总 chunks:")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass

        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "purged": purged,
            "total": groups,
            "duration_s": duration,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout (>10min)", "duration_s": time.time() - t0}
    except (OSError, RuntimeError, ValueError) as e:
        return {"ok": False, "reason": str(e), "duration_s": time.time() - t0}


def run_once(verbose: bool = False, dry_run: bool = False, skip_distill: bool = False) -> int:
    """跑一次 ingest + 蒸馏兜底，返回 0（成功）或 1（失败但不影响 WorkBuddy 启动）"""
    try:
        _log(f"=== 启动 ingest_wb_memory_oneshot (dry_run={dry_run}, skip_distill={skip_distill}) ===", verbose)

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

        # 健康度快照
        total_chunks = 0
        try:
            health = mem.indexer.all_chunks()
            total_chunks = len(health)
            _log(f"索引现状: total={total_chunks} chunks", verbose)
        except (OSError, RuntimeError, ValueError) as e:
            _log(f"健康度快照失败: {e}", verbose)

        # 蒸馏兜底（v0.2.4）：解决电脑不开机时 03:00 cron 漏跑的问题
        if not skip_distill and not dry_run and total_chunks > 0:
            try:
                state = _load_distill_state(index_dir)
                should_run, reason = _should_distill(state, current_chunks=total_chunks)
                if should_run:
                    _log(f"🔧 触发蒸馏兜底: {reason}", verbose)
                    d_result = _run_distill(index_dir, verbose=verbose)
                    if d_result.get("ok"):
                        _log(
                            f"✅ 蒸馏完成: purged={d_result['purged']} "
                            f"duration={d_result['duration_s']:.1f}s",
                            verbose,
                        )
                        # 重新读索引大小，更新状态
                        try:
                            new_total = len(mem.indexer.all_chunks())
                            _save_distill_state(index_dir, new_total)
                        except (OSError, RuntimeError, ValueError):
                            _save_distill_state(index_dir, total_chunks)
                    else:
                        _log(
                            f"⚠️ 蒸馏失败（已忽略）: {d_result.get('reason', '?')}",
                            verbose,
                        )
                else:
                    _log(f"⏭️  蒸馏跳过: {reason}", verbose)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                _log(f"蒸馏兜底异常（已忽略）: {e}", verbose)

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
    parser.add_argument("--skip-distill", action="store_true",
                        help="跳过蒸馏兜底（默认会检查并按需补跑）")
    args = parser.parse_args()

    sys.exit(run_once(
        verbose=args.verbose,
        dry_run=args.dry_run,
        skip_distill=args.skip_distill,
    ))


if __name__ == "__main__":
    main()