"""
reclassify_unknown.py — 重新分类 source='unknown' 的 chunks

v0.2.5 之前 ingest 路径通过 detect_source() 兜底把所有"无法识别 source_path"
的合法内容打成 'unknown'。这不是污染，是误标。

判定规则（优先级从高到低）：
1. text 含 `<!-- RAW_JSON_START` 残留        → source='cloud-profile-legacy'
   (保留数据但打标，方便排查)
2. text 以 **xxx** 开头（MEMORY.md 风格）    → source='user-memory'
3. text 含 `YYYY-MM-DD` 日期 + 决策/记录词   → source='workspace-log'
4. text 含技术决策关键词（决定/弃用/采用）     → source='user-memory'
5. text 含测试验证内容（验证/召回/实测/通过）  → source='test-result'
6. 都不匹配                                 → source='user-memory'（兜底改好）

策略：默认兜底改成 'user-memory'（保守标记，宁可让来源 user-memory 也不标 unknown）
      未来 detect_source() 修好后，新 ingest 自动正确，无需再跑此脚本。

用法：
  python -m scripts.reclassify_unknown --dry-run    # 只报告，不改
  python -m scripts.reclassify_unknown             # 实际执行
  python -m scripts.reclassify_unknown --verbose   # 每条都打
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_index_dir  # noqa: E402


# 决策词：决定 / 弃用 / 采用 / 不用 / 失败 / 成功 / 选用 / 选择
DECISION_KW = ("决定", "弃用", "采用", "不用", "失败", "成功", "选用", "选择", "禁用", "改用")
# 测试/验证关键词
TEST_KW = ("验证", "召回", "实测", "测试通过", "pytest", "PASSED", "FAILED")
# MEMORY.md 风格开头：**xxx**
MEMORY_HEAD = re.compile(r"^\*\*[^*]+\*\*")


def classify(text: str) -> tuple[str, str]:
    """
    根据文本特征返回 (新 source, 判定理由)。
    """
    head = text.lstrip()[:200]

    # 1. RAW_JSON 残留
    if "<!-- RAW_JSON_START" in text or text.lstrip().startswith("RAW_JSON_END"):
        return "cloud-profile-legacy", "contains RAW_JSON markers"

    # 2. MEMORY.md 风格 H1/H2
    if MEMORY_HEAD.match(head):
        return "user-memory", "starts with **xxx** (MEMORY.md style)"

    # 3. 含日期 + 决策词 → workspace-log
    has_date = bool(re.search(r"\d{4}-\d{2}-\d{2}", text))
    has_decision = any(kw in text for kw in DECISION_KW)
    if has_date and has_decision:
        return "workspace-log", "date + decision keyword"

    # 4. 含决策词（即使无日期）→ user-memory（决策通常是用户偏好/项目决定）
    if has_decision:
        return "user-memory", "decision keyword present"

    # 5. 含测试/验证关键词 → test-result
    if any(kw in text for kw in TEST_KW):
        return "test-result", "test/verification keyword"

    # 6. 兜底：unknown → user-memory（保守标记）
    return "user-memory", "fallback (conservative)"


def find_unknown(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        con.execute(
            """
            SELECT id, length, access_count, substr(text, 1, 100) AS head
            FROM chunks
            WHERE source = 'unknown'
            ORDER BY access_count DESC, length DESC
            """
        )
    )


def reclassify(con: sqlite3.Connection, ids: list[str], new_sources: dict[str, str]) -> int:
    """按 id 批量 update source"""
    if not ids:
        return 0
    cur = con.executemany(
        "UPDATE chunks SET source = ? WHERE id = ?",
        [(new_sources[i], i) for i in ids],
    )
    con.commit()
    return cur.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="只报告，不改")
    parser.add_argument("--verbose", action="store_true", help="每条都打")
    args = parser.parse_args()

    index_dir = Path(get_index_dir())
    db_path = index_dir / "meta.db"
    if not db_path.exists():
        raise FileNotFoundError(f"meta.db not found: {db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # 先取 text（--verbose 才用）
    rows = list(
        con.execute(
            "SELECT id, text FROM chunks WHERE source = 'unknown'"
        )
    )

    # 分桶
    decisions: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for r in rows:
        new_src, reason = classify(r["text"])
        decisions[r["id"]] = new_src
        reasons[r["id"]] = reason

    counts = Counter(decisions.values())
    print(f"Reclassify plan for {len(rows)} source='unknown' chunks in {db_path}:")
    for src, n in counts.most_common():
        print(f"  → {src}: {n}")
    print()

    if args.verbose:
        # 按新 source 分组打
        by_new: dict[str, list[tuple[str, str]]] = {}
        for rid, new_src in decisions.items():
            by_new.setdefault(new_src, []).append((rid, reasons[rid]))
        for new_src, items in by_new.items():
            print(f"--- → {new_src} ({len(items)}) ---")
            for rid, reason in items[:5]:
                print(f"  id={rid[:28]:<30}  reason={reason}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
            print()

    if args.dry_run:
        print(f"[DRY-RUN] Would UPDATE {len(rows)} chunks. Run without --dry-run to execute.")
        return

    n = reclassify(con, list(decisions.keys()), decisions)
    print(f"✅ Reclassified {n} chunks.")
    print()
    print("Verify with:")
    print("  python -m scripts.health")


if __name__ == "__main__":
    main()
