"""
评估脚本：Recall@K / MRR / NDCG@K
- 读取 data/gold_set.jsonl
- 逐条 query 检索
- 计算指标 + 输出报告

gold_set.jsonl 格式:
{"query": "...", "relevant_ids": ["chunk_xxx", ...]}
"""
import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory


def load_gold_set(path: Path) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hit = len([i for i in top_k if i in relevant_ids])
    return hit / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    import math
    top_k = retrieved_ids[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(top_k) if rid in relevant_ids)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="评估 RAG 检索质量")
    parser.add_argument("--gold", default="./data/gold_set.jsonl")
    parser.add_argument("--index-dir", default="./.index")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true", help="启用 BGE Rerank")
    parser.add_argument("--candidates", type=int, default=20, help="候选池大小")
    parser.add_argument("--report", default="./data/eval_report.md")
    args = parser.parse_args()

    gold = load_gold_set(Path(args.gold))
    if not gold:
        print(f"❌ gold set 为空: {args.gold}")
        sys.exit(1)

    print(f"📋 Gold set: {len(gold)} 条 query")

    # 可选 Reranker
    reranker = None
    if args.rerank:
        from src.reranker import Reranker
        reranker = Reranker()

    mem = Memory(reranker=reranker)
    print(f"🔧 索引: total={mem.indexer.count()} | rerank={args.rerank}\n")

    metrics = defaultdict(list)
    latencies = []
    details = []

    for item in gold:
        query = item["query"]
        relevant = set(item["relevant_ids"])

        t0 = time.perf_counter()
        results = mem.search(query, top_k=args.top_k, candidates=args.candidates, rerank=args.rerank)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        retrieved_ids = [r.id for r in results]
        r1 = recall_at_k(retrieved_ids, relevant, 1)
        r5 = recall_at_k(retrieved_ids, relevant, 5)
        r10 = recall_at_k(retrieved_ids, relevant, 10)
        m = mrr(retrieved_ids, relevant)
        n = ndcg_at_k(retrieved_ids, relevant, 10)

        metrics["recall@1"].append(r1)
        metrics["recall@5"].append(r5)
        metrics["recall@10"].append(r10)
        metrics["mrr"].append(m)
        metrics["ndcg@10"].append(n)

        details.append({
            "query": query,
            "relevant": len(relevant),
            "retrieved": len(retrieved_ids),
            "recall@5": round(r5, 3),
            "mrr": round(m, 3),
            "latency_ms": round(latency, 1),
        })

        status = "✅" if r5 > 0 else "❌"
        print(f"  {status} {query[:50]:<50}  R@5={r5:.2f}  MRR={m:.2f}  {latency:.0f}ms")

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"📊 评估结果 (n={len(gold)})")
    print(f"{'=' * 60}")
    summary = {}
    for k, vs in metrics.items():
        avg = sum(vs) / len(vs) if vs else 0
        summary[k] = round(avg, 4)
        print(f"  {k:<12} {avg:.4f}")
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
    print(f"  {'latency_p50':<12} {p50:.1f}ms")
    print(f"  {'latency_p95':<12} {p95:.1f}ms")
    summary["latency_p50_ms"] = round(p50, 1)
    summary["latency_p95_ms"] = round(p95, 1)

    # 写报告
    report = ["# RAG 评估报告", "", f"**评估时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    report.append(f"**Gold set 数量**: {len(gold)}  ")
    report.append(f"**索引 chunk 总数**: {mem.indexer.count()}")
    report.append("")
    report.append("## 总览指标")
    report.append("")
    report.append("| 指标 | 数值 |")
    report.append("|---|---|")
    for k, v in summary.items():
        report.append(f"| {k} | {v} |")
    report.append("")
    report.append("## 逐条详情")
    report.append("")
    report.append("| Query | 相关数 | 召回数 | Recall@5 | MRR | 延迟 |")
    report.append("|---|---|---|---|---|---|")
    for d in details:
        report.append(f"| {d['query'][:40]} | {d['relevant']} | {d['retrieved']} | {d['recall@5']} | {d['mrr']} | {d['latency_ms']}ms |")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report), encoding="utf-8")
    print(f"\n📝 报告已保存: {args.report}")


if __name__ == "__main__":
    main()
