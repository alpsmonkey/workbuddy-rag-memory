"""
清噪声 + 去重脚本
- 按 text hash 去重（保留最早 ts 版本）
- 按 project 白名单过滤（保留核心 RAG 记忆）
- 删除满足条件的 chunk

用法:
    python scripts/dedup_noise.py --index-dir <DIR> [--dry-run] [--keep-projects ...]
"""
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexer import Indexer


# 核心 RAG 记忆项目白名单（保留）
DEFAULT_KEEP_PROJECTS: Set[str] = {
    # 用户/身份/工作习惯
    "鹏哥偏好", "用户身份", "工作习惯", "代码",
    # 项目级用户记忆
    "User-level", "工作区记忆", "历史只能",
    # RAG 系统自身
    "RAG", "阶段", "路径", "交付清单", "关键设计决策",
    "名启发式提取精度有限", "端到端唯一标记",
    # 设计决策
    "技术决策", "技术选型", "优化目标", "评分维度",
    "去重策略", "时间衰减", "背景",
    # SkillFather 核心
    "SkillFather", "WorkBuddy",
}

# 噪声项目（按需补充）
NOISE_PROJECTS: Set[str] = {
    "cheat-on-content", "SOP", "USB加密狗推广视频", "Cover",
    "baoyu-xhs-images", "Excel", "小红书爆款", "Humanizer",
    "Hash", "Blind", "Part1", "Spearman", "Superpowers",
    "URL", "第一篇稿子", "第二篇稿子", "第3篇稿子", "第4篇稿子",
    "写稿", "对标账号分析", "复盘", "校准池状态", "核心发现",
    "对比优化", "文章优化", "Prompt", "脚手架", "STATUS.md",
    "README.md", "配置修改", "发布记录", "记忆",
}


def _normalize(text: str) -> str:
    """轻归一化用于 hash"""
    return " ".join(text.split())


def _text_hash(text: str) -> str:
    return hashlib.md5(_normalize(text).encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="清噪声 + 去重")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--dry-run", action="store_true", help="只统计不删除")
    parser.add_argument("--keep-projects", nargs="*", default=None,
                        help="覆盖默认白名单")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    keep = set(args.keep_projects) if args.keep_projects else DEFAULT_KEEP_PROJECTS

    idx = Indexer(args.index_dir)
    all_chunks = idx.all_chunks()
    print(f"📦 起始 chunks: {len(all_chunks)}")

    # 按 text hash 分组
    by_hash: Dict[str, List[dict]] = {}
    for c in all_chunks:
        h = _text_hash(c.get("text", ""))
        by_hash.setdefault(h, []).append(c)

    # 每组保留 ts 最小（最早）的
    canonical: Dict[str, dict] = {}
    duplicates: List[str] = []
    for h, group in by_hash.items():
        if len(group) == 1:
            canonical[h] = group[0]
        else:
            group.sort(key=lambda x: x.get("ts", "") or "")
            canonical[h] = group[0]
            duplicates.extend(c["id"] for c in group[1:])

    print(f"🧬 唯一 text: {len(canonical)}")
    print(f"♻️  重复待删: {len(duplicates)}")

    # 按 project 白名单过滤
    keep_ids: Set[str] = set()
    drop_by_project: List[str] = []
    for h, c in canonical.items():
        project = c.get("project", "") or ""
        if project in keep:
            keep_ids.add(c["id"])
        else:
            drop_by_project.append(c["id"])

    print(f"✅ 保留 (project 白名单): {len(keep_ids)}")
    print(f"❌ 删 (noise project): {len(drop_by_project)}")
    print(f"🗑️  待删总数: {len(duplicates) + len(drop_by_project)}")

    if args.verbose:
        print(f"\n保留项目: {sorted(keep)}")
        print(f"噪声项目样本: {sorted(NOISE_PROJECTS)[:10]}...")

    if args.dry_run:
        print("\n[DRY-RUN] 不实际删除")
        return

    # 实际删除
    all_drop = set(duplicates) | set(drop_by_project)
    deleted = 0
    for cid in all_drop:
        try:
            if idx.delete(cid):
                deleted += 1
        except Exception as e:
            if args.verbose:
                print(f"  ⚠️  {cid}: {e}")

    print(f"\n✓ 实际删除: {deleted}")
    print(f"📦 剩余: {idx.count()}")


if __name__ == "__main__":
    main()