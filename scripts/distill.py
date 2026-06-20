"""
阶段 3：自动蒸馏
- 按 project / source 分组
- 按 access_count 排序（热记忆优先）
- 应用时间衰减，自动清理低价值旧记忆
- 输出汇总报告 + 候选保留清单

用法:
  python -m scripts.distill                    # 蒸馏 + 报告
  python -m scripts.distill --project skillfather  # 只看某个项目
  python -m scripts.distill --decay            # 清理低价值旧记忆
  python -m scripts.distill --dry-run          # 只看候选，不改索引
  python -m scripts.distill --json             # 机器可读
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory
from src.retriever import _parse_ts


# 默认时间衰减常数（和 Retriever 保持一致）
DEFAULT_TAU_DAYS = 90.0

# 默认保留阈值：低于此分数的视为"可清理"
DEFAULT_PURGE_THRESHOLD = 0.005


def compute_decay_score(chunk: dict, tau_days: float = DEFAULT_TAU_DAYS) -> float:
    """
    蒸馏用的综合分数
    score = rrf_proxy × exp(-Δt/τ) × log(1+access_count)

    rrf_proxy 用 confidence 代替（蒸馏阶段没有 query）
    """
    conf = float(chunk.get("confidence", 0.5))
    ts = _parse_ts(chunk.get("ts", ""))
    access = int(chunk.get("access_count", 0))

    if ts:
        delta_days = (datetime.now() - ts).total_seconds() / 86400.0
        decay = math.exp(-delta_days / tau_days)
    else:
        decay = 0.5  # 无时间戳默认中等

    pop = math.log1p(access + 1)

    return conf * decay * pop


def distill(
    mem: Memory,
    project: str = None,
    top_per_group: int = 10,
    purge_threshold: float = DEFAULT_PURGE_THRESHOLD,
    dry_run: bool = False,
) -> dict:
    """
    主入口

    Returns:
        {
            "groups": {
                "<project>": {
                    "total": int,
                    "candidates_kept": [...],   # 高分记忆
                    "candidates_purge": [...],  # 低分待清理
                }
            },
            "summary": {
                "total_chunks": int,
                "purged": int,
                "avg_score": float,
            }
        }
    """
    chunks = mem.indexer.all_chunks()
    if not chunks:
        return {
            "groups": {},
            "summary": {"total_chunks": 0, "purged": 0, "avg_score": 0.0},
            "note": "index is empty",
        }

    # 按 project 分组（缺失归到 "(未分类)"）
    groups: dict = defaultdict(list)
    for c in chunks:
        if project and (c.get("project") or "(未分类)") != project:
            continue
        key = c.get("project") or "(未分类)"
        groups[key].append(c)

    # 对每个组算分 + 排序
    out_groups = {}
    all_purged = []
    score_sum = 0.0
    n_scored = 0

    for proj_key, items in groups.items():
        scored = []
        for c in items:
            s = compute_decay_score(c)
            c["_decay_score"] = s
            scored.append(c)
            score_sum += s
            n_scored += 1

        scored.sort(key=lambda x: x["_decay_score"], reverse=True)
        keep = scored[:top_per_group]
        purge = [c for c in scored[top_per_group:] if c["_decay_score"] < purge_threshold]

        out_groups[proj_key] = {
            "total": len(items),
            "kept_count": len(keep),
            "purge_count": len(purge),
            "kept_top": [
                {
                    "id": c["id"],
                    "text": c["text"][:120],
                    "score": round(c["_decay_score"], 4),
                    "access": c.get("access_count", 0),
                    "ts": c.get("ts", ""),
                    "project": c.get("project", ""),
                }
                for c in keep
            ],
            "purge_candidates": [
                {
                    "id": c["id"],
                    "text": c["text"][:80],
                    "score": round(c["_decay_score"], 4),
                    "access": c.get("access_count", 0),
                    "ts": c.get("ts", ""),
                }
                for c in purge[:20]  # 报告只列前 20
            ],
        }
        all_purged.extend(purge)

    # 实际清理
    purged_count = 0
    if purge and not dry_run:
        for c in all_purged:
            # 走 dedup 的方式不可逆删除，用 sqlite 直接 DELETE
            try:
                from src.indexer import Indexer as _I
                # 临时复用 _conn 模式删除
                with mem.indexer._conn() as conn:
                    conn.execute("DELETE FROM chunks WHERE id=?", (c["id"],))
                    conn.execute("DELETE FROM chunks_fts WHERE id=?", (c["id"],))
                    conn.commit()
                # LanceDB 删除
                try:
                    mem.indexer.table.delete(f"id = '{c['id']}'")
                except Exception:
                    pass
                purged_count += 1
            except Exception as e:
                print(f"[warn] 删除失败 {c['id']}: {e}")

    return {
        "groups": out_groups,
        "summary": {
            "total_chunks": len(chunks),
            "projects": len(groups),
            "purged": purged_count,
            "purge_threshold": purge_threshold,
            "avg_score": round(score_sum / n_scored, 4) if n_scored else 0,
            "dry_run": dry_run,
        },
    }


def print_human(report: dict, verbose: bool = False) -> None:
    s = report["summary"]
    print()
    print("=" * 70)
    print(f"  WorkBuddy RAG 蒸馏报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总 chunks: {s['total_chunks']} | 项目: {s['projects']} | 平均分: {s['avg_score']}")
    print(f"  清理阈值: {s['purge_threshold']} | 实际清理: {s['purged']} (dry_run={s['dry_run']})")
    print("=" * 70)

    for proj, info in report["groups"].items():
        print(f"\n📁 {proj}  (total={info['total']}, keep={info['kept_count']}, purge={info['purge_count']})")
        if verbose or info["kept_count"] > 0:
            print(f"   ⭐ Top {info['kept_count']} 保留:")
            for c in info["kept_top"]:
                print(f"      {c['score']:.4f}  acc={c['access']:>3}  {c['text']}")
        if info["purge_candidates"]:
            print(f"   🗑️  待清理 (展示前 {len(info['purge_candidates'])}):")
            for c in info["purge_candidates"]:
                print(f"      {c['score']:.4f}  acc={c['access']:>3}  {c['text']}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="自动蒸馏 RAG 索引")
    parser.add_argument("--index-dir", default="./.index")
    parser.add_argument("--project", default=None, help="只蒸馏某个项目")
    parser.add_argument("--top", type=int, default=10, help="每组保留 top N")
    parser.add_argument("--threshold", type=float, default=DEFAULT_PURGE_THRESHOLD,
                        help="清理阈值（低于此分视为低价值）")
    parser.add_argument("--dry-run", action="store_true", help="只看不改")
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示完整 top")
    args = parser.parse_args()

    mem = Memory(index_dir=args.index_dir)
    print(f"🔧 {mem}", file=sys.stderr)

    report = distill(
        mem,
        project=args.project,
        top_per_group=args.top,
        purge_threshold=args.threshold,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report, args.verbose)


if __name__ == "__main__":
    main()