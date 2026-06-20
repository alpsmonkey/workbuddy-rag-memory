"""
导入 MEMORY.md / 日志文件
用法: python -m scripts.ingest <file_path> [--project NAME]
"""
import argparse
import sys
from pathlib import Path

# 允许从项目根直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory


def main():
    parser = argparse.ArgumentParser(description="导入记忆文件")
    parser.add_argument("file", help="待导入文件路径（MEMORY.md / 日志等）")
    parser.add_argument("--project", default=None, help="项目名（覆盖启发式提取）")
    parser.add_argument("--index-dir", default="./.index", help="索引目录")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    print(f"📄 读取: {path} ({len(text)} 字符)")

    mem = Memory(index_dir=args.index_dir)
    print(f"🔧 {mem}")

    if args.project:
        # 项目名注入：包装为 chunk
        from src.chunker import chunk_text
        chunks = chunk_text(text, source=str(path))
        for c in chunks:
            c.meta["project"] = args.project
        results = mem.add_chunks(chunks)
    else:
        result = mem.add(text, source=str(path))
        results = [result]

    # 统计
    from collections import Counter
    decisions = Counter(r.decision.value for r in results)
    print(f"\n✅ 导入完成: {len(results)} chunks")
    for k, v in decisions.items():
        print(f"   - {k}: {v}")

    if args.verbose:
        for r in results:
            print(f"\n  [{r.decision.value}] sim={r.similarity:.3f}  {r.reason}")
            if r.existing_id:
                print(f"    冲突 ID: {r.existing_id}")

    print(f"\n📊 索引现状: total={mem.indexer.count()}")


if __name__ == "__main__":
    main()
