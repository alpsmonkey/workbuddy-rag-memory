"""
命令行检索
用法: python -m scripts.query "上次那个方案" [--top-k 5] [--project NAME]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory


def main():
    parser = argparse.ArgumentParser(description="检索记忆")
    parser.add_argument("query", help="查询文本")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--project", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--index-dir", default="./.index")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    mem = Memory(index_dir=args.index_dir)
    print(f"🔍 查询: {args.query}\n")

    results = mem.search(args.query, top_k=args.top_k, project=args.project, source=args.source)

    if not results:
        print("❌ 无结果")
        return

    if args.format == "json":
        import json
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
    else:
        for i, r in enumerate(results, 1):
            print(f"{'─' * 60}")
            print(f"#{i}  score={r.score:.4f}  conf={r.confidence:.2f}")
            print(f"    src: {r.source}  project: {r.project or '(无)'}")
            print(f"    ts:  {r.ts}")
            if r.entities:
                print(f"    tags: {', '.join(r.entities[:5])}")
            print(f"\n    {r.text[:300]}{'...' if len(r.text) > 300 else ''}")
        print(f"\n{'─' * 60}")
        print(f"共 {len(results)} 条结果")


if __name__ == "__main__":
    main()
