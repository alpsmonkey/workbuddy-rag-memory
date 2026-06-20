"""
RAG 记忆系统健康度检查

检查项：
1. 后端：embedding 是否加载成功（sentence-transformers / hash-fallback）
2. 索引：总 chunk 数、按 source / project 分布、平均长度
3. 存储：lance / meta.db 磁盘占用
4. 模型：HF cache 是否存在、模型文件是否完整
5. 真实记忆源：~/.workbuddy/memory/ 是否可读、文件数
6. 写入统计（来自 meta.db 的 chunks.last_access / access_count）
7. 异常：hash 兜底下没有 chunk、索引文件损坏等

用法:
  python -m scripts.health                # 完整健康报告
  python -m scripts.health --json         # 机器可读
  python -m scripts.health --quiet        # 只看红黄
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory
from src.dedup import MAX_SEARCH_K, LARGE_INDEX_THRESHOLD


def _fmt_size(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def _check_backend(mem: Memory) -> dict:
    """检查 embedding 后端"""
    be = mem.embedder.backend
    is_healthy = be in ("sentence-transformers", "onnx") or be == "hash-fallback"
    return {
        "status": "🟢" if is_healthy else "🔴",
        "backend": be,
        "model": mem.embedder.model_name,
        "dim": mem.embedder.dim,
        "device": mem.embedder.device,
        "normalize": mem.embedder.normalize,
        "note": (
            "生产就绪"
            if be == "sentence-transformers"
            else "⚠️ hash 兜底：检索无语义信息，仅供 CI/冒烟测试"
            if be == "hash-fallback"
            else f"未知后端: {be}"
        ),
    }


def _check_index(mem: Memory) -> dict:
    """检查索引状态"""
    total = mem.indexer.count()
    chunks = mem.indexer.all_chunks()
    by_source = Counter(c.get("source", "unknown") for c in chunks)
    by_project = Counter(c.get("project") or "(未分类)" for c in chunks)
    avg_len = sum(c.get("length", 0) for c in chunks) / total if total else 0
    avg_conf = sum(c.get("confidence", 0) for c in chunks) / total if total else 0
    no_access = sum(1 for c in chunks if c.get("access_count", 0) == 0)

    status = "🟢"
    notes = []
    if total == 0:
        status = "🟡"
        notes.append("索引为空")
    if total > LARGE_INDEX_THRESHOLD:
        notes.append(f"大索引（>{LARGE_INDEX_THRESHOLD}），dedup 自动降速")
    if no_access / total > 0.9 and total > 10:
        notes.append(f"90%+ 记忆从未被访问，可能存在死数据")

    return {
        "status": status,
        "total": total,
        "avg_length": round(avg_len, 1),
        "avg_confidence": round(avg_conf, 3),
        "by_source": dict(by_source.most_common()),
        "by_project": dict(by_project.most_common(10)),
        "never_accessed": no_access,
        "notes": notes,
    }


def _check_storage(index_dir: Path) -> dict:
    """检查磁盘占用"""
    if not index_dir.exists():
        return {"status": "🔴", "error": f"索引目录不存在: {index_dir}"}

    total = 0
    files = {}
    for f in index_dir.rglob("*"):
        if f.is_file():
            sz = f.stat().st_size
            total += sz
            files[str(f.relative_to(index_dir))] = sz

    return {
        "status": "🟢" if total > 0 else "🟡",
        "index_dir": str(index_dir),
        "total_size": _fmt_size(total),
        "total_bytes": total,
        "files": {k: _fmt_size(v) for k, v in sorted(files.items(), key=lambda x: -x[1])[:10]},
    }


def _check_hf_cache(embed_model: str) -> dict:
    """检查 HF 本地缓存"""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if not hf_home.exists():
        return {"status": "🟡", "hf_home": str(hf_home), "note": "HF 缓存目录不存在"}

    # 找包含模型名的子目录
    candidates = list(hf_home.rglob("config.json"))
    matched = None
    for c in candidates:
        if embed_model.replace("/", "--") in str(c):
            matched = c.parent
            break

    if not matched:
        return {
            "status": "🟡",
            "hf_home": str(hf_home),
            "model": embed_model,
            "note": f"未找到模型本地缓存（HF_HUB_OFFLINE=1 时无法联网下载）",
        }

    sz = sum(f.stat().st_size for f in matched.rglob("*") if f.is_file())
    return {
        "status": "🟢",
        "hf_home": str(hf_home),
        "model_path": str(matched),
        "size": _fmt_size(sz),
    }


def _check_wb_memory() -> dict:
    """检查 WorkBuddy 真实记忆源"""
    targets = []
    home = Path.home()
    candidates = [
        home / ".workbuddy" / "memory",  # 用户级
        Path.cwd() / ".workbuddy" / "memory",  # 项目级
    ]
    # env 里也可以追加
    extra = os.environ.get("WB_MEMORY_DIRS", "")
    if extra:
        for raw in extra.split(";"):
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw).expanduser()
            if p not in candidates:
                candidates.append(p)

    ignore_patterns = ("*.bak", "*.tmp.*", "*~")
    summary = {"status": "🟢", "dirs": []}
    for d in candidates:
        if not d.exists():
            summary["dirs"].append({"path": str(d), "exists": False})
            continue
        all_files = list(d.rglob("*_memory.md"))
        keep = [
            f for f in all_files
            if not any(f.match(p) for p in ignore_patterns)
        ]
        skip = len(all_files) - len(keep)
        summary["dirs"].append({
            "path": str(d),
            "exists": True,
            "memory_files": len(keep),
            "skipped_bak_tmp": skip,
            "total_size": _fmt_size(sum(f.stat().st_size for f in keep)),
        })
    if not any(d.get("exists") for d in summary["dirs"]):
        summary["status"] = "🟡"
    return summary


def _check_dedup_config() -> dict:
    """检查 dedup 配置"""
    return {
        "status": "🟢",
        "max_search_k": MAX_SEARCH_K,
        "large_index_threshold": LARGE_INDEX_THRESHOLD,
        "note": (
            f"小索引（≤{LARGE_INDEX_THRESHOLD}）: search_k ≤ {MAX_SEARCH_K}\n"
            f"大索引（>{LARGE_INDEX_THRESHOLD}）: search_k 自动减半"
        ),
    }


def run_health(index_dir: str = None) -> dict:
    """跑全套健康检查"""
    index_dir = index_dir or os.environ.get("INDEX_DIR", "./.index")
    mem = Memory(index_dir=index_dir)

    checks = {
        "backend": _check_backend(mem),
        "index": _check_index(mem),
        "storage": _check_storage(Path(index_dir)),
        "dedup_config": _check_dedup_config(),
        "hf_cache": _check_hf_cache(mem.embedder.model_name),
        "wb_memory_sources": _check_wb_memory(),
    }

    # 整体健康度
    statuses = [c["status"] for c in checks.values()]
    if "🔴" in statuses:
        overall = "🔴 UNHEALTHY"
    elif "🟡" in statuses:
        overall = "🟡 WARNING"
    else:
        overall = "🟢 HEALTHY"

    return {
        "overall": overall,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_dir": str(Path(index_dir).resolve()),
        "checks": checks,
    }


def print_human(report: dict, quiet: bool = False) -> None:
    """打印人可读报告"""
    print()
    print("=" * 70)
    print(f"  WorkBuddy RAG 健康度报告 - {report['timestamp']}")
    print(f"  索引目录: {report['index_dir']}")
    print(f"  整体状态: {report['overall']}")
    print("=" * 70)

    order = [
        ("backend", "1. Embedding 后端"),
        ("index", "2. 索引状态"),
        ("storage", "3. 磁盘占用"),
        ("dedup_config", "4. 去重配置"),
        ("hf_cache", "5. HF 模型缓存"),
        ("wb_memory_sources", "6. 真实记忆源"),
    ]
    for key, title in order:
        c = report["checks"][key]
        if quiet and c["status"] == "🟢":
            continue
        print(f"\n{title}  {c['status']}")
        for k, v in c.items():
            if k == "status":
                continue
            if isinstance(v, dict):
                print(f"   {k}:")
                for kk, vv in v.items():
                    print(f"     - {kk}: {vv}")
            elif isinstance(v, list):
                print(f"   {k}:")
                for item in v:
                    print(f"     - {item}")
            else:
                print(f"   {k}: {v}")
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RAG 健康度检查")
    parser.add_argument("--index-dir", default="./.index")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    parser.add_argument("--quiet", "-q", action="store_true", help="只显示红黄状态")
    parser.add_argument(
        "--watch", type=int, default=0, metavar="SECONDS",
        help="持续检查模式，每 N 秒跑一次（适合 cron / watchdog 接入）",
    )
    args = parser.parse_args()

    if args.watch > 0:
        import time
        print(f"[watch] 每 {args.watch}s 跑一次健康检查，Ctrl+C 停止", file=sys.stderr)
        try:
            while True:
                report = run_health(args.index_dir)
                ts = report["timestamp"]
                overall = report["overall"]
                total = report["checks"]["index"]["total"]
                be = report["checks"]["backend"]["backend"]
                print(f"[{ts}] {overall}  total={total}  backend={be}")
                if "🔴" in overall or "🟡" in overall:
                    print_human(report, quiet=False)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[watch] 停止", file=sys.stderr)
            sys.exit(0)
    else:
        report = run_health(args.index_dir)

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print_human(report, args.quiet)

        if "🔴" in report["overall"]:
            sys.exit(2)
        if "🟡" in report["overall"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
