"""
扫描 ~/.workbuddy/memory/ 和 ./.workbuddy/memory/ 并入库

用法:
  python -m scripts.ingest_wb_memory                          # 默认目录
  python -m scripts.ingest_wb_memory --dir ~/foo --dir ./bar # 自定义目录
  python -m scripts.ingest_wb_memory --project workbuddy     # 强制打 project
  python -m scripts.ingest_wb_memory --dry-run               # 只看会扫到哪些
  python -m scripts.ingest_wb_memory --no-incremental        # 强制全量重扫
  python -m scripts.ingest_wb_memory --json                  # 机器可读
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory, DEFAULT_WB_MEMORY_DIRS, DEFAULT_IGNORE_PATTERNS


def main():
    parser = argparse.ArgumentParser(description="扫描 WorkBuddy 真实记忆目录入库")
    parser.add_argument(
        "--dir", action="append", default=None,
        help="记忆目录（可多次指定，默认 ~/.workbuddy/memory + ./.workbuddy/memory）",
    )
    parser.add_argument(
        "--ignore", action="append", default=None,
        help="忽略 glob 模式（可多次指定）",
    )
    parser.add_argument("--project", default=None, help="强制 project 标签")
    parser.add_argument("--index-dir", default="./.index", help="索引目录")
    parser.add_argument("--dry-run", action="store_true", help="只扫描不入库")
    parser.add_argument("--no-incremental", action="store_true",
                        help="禁用增量（强制全量重扫）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    dirs = args.dir if args.dir else DEFAULT_WB_MEMORY_DIRS
    ignore = args.ignore if args.ignore else DEFAULT_IGNORE_PATTERNS
    incremental = not args.no_incremental

    print(f"📁 扫描目录: {dirs}")
    print(f"🚫 忽略模式: {ignore}")
    print(f"⚙️  增量模式: {'ON' if incremental else 'OFF'}")
    print()

    mem = Memory(index_dir=args.index_dir)
    print(f"🔧 {mem}")
    print()

    report = mem.scan_workbuddy_memory(
        dirs=dirs,
        ignore=ignore,
        project=args.project,
        incremental=incremental,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if report.get("note"):
            print(f"⚠️  {report['note']}")
            return

        print(f"📊 扫描: {report['scanned']} 个文件，忽略 {report['skipped']} 个")
        if incremental and not args.dry_run:
            print(f"⏭️  未变化: {report['unchanged']} 个文件（增量跳过）")
        if report["chunks_total"]:
            print(f"📦 切出 chunk: {report['chunks_total']} 条")
            print(f"   - insert (新增):     {report['inserted']}")
            print(f"   - merge  (合并):     {report['merged']}")
            print(f"   - skip   (去重跳过): {report['skipped_dup']}")
        elif report["unchanged"] and incremental:
            print(f"✨ 所有文件未变化，无需处理")

        if args.verbose and report["files"]:
            print("\n📂 逐文件详情:")
            for fi in report["files"]:
                path = Path(fi["path"])
                if args.dry_run:
                    sz = fi.get("size", 0)
                    print(f"   {path.name:<40}  {sz} B  (待入库)")
                else:
                    decs = ", ".join(f"{k}={v}" for k, v in fi.get("decisions", {}).items()) or "(无)"
                    print(f"   {path.name:<40}  {fi.get('chunks', 0):>3} chunks  [{decs}]")

        print()
        print(f"📈 索引现状: total={mem.indexer.count()}")


if __name__ == "__main__":
    main()