"""
D 方案: 蒸馏 → 自动注入到 ~/.workbuddy/MEMORY.md (L2 user-level)

工作流:
  1. 调 Memory + distill() 拿每个 project 的 kept_top (高分骨架)
  2. 读 ~/.workbuddy/MEMORY.md 现状
  3. 用标记区间 <!-- AUTO_INJECT_START --> ... <!-- AUTO_INJECT_END --> 隔离用户手写 + 自动注入
  4. 替换自动注入区内容（每次重写，最新分数）
  5. 字符预算保护：手写区不动，自动注入区 ≤ 2400 字符（占 L2 总预算 4000 的 60%）

为什么用 distill() 输出而不是 RAG.search()？
  - distill 已经按 access_count + 时间衰减 + confidence 综合排序，最稳
  - RAG.search() 需 query，启动时无 query 上下文
  - distill 输出可解释（每条都带 score + access + ts）

为什么只在标记区间内重写？
  - 鹏哥手写的 6 行偏好（Python 3.13 / 5 个一批 / 表格 > 段落...）不能动
  - 标记区间让"自动注入"成为可重入的：每次都覆盖，不留历史污染
  - 用户可手动删 AUTO_INJECT_END 后面的内容，停止自动注入

触发点（与 ingest_wb_memory_oneshot.py 配套）：
  1. 蒸馏兜底后（自动）—— 在 ingest_wb_memory_oneshot.py 中串联
  2. 独立 CLI（手动）—— `python -m scripts.inject_distilled_to_memory`
  3. 未来可加每日 cron 04:00（与 distill 错开 1h）

字符预算（防御性）：
  - L2 MEMORY.md 总预算: 4000 字符
  - 手写区: 不可控（鹏哥自管）
  - 自动注入区: 硬上限 2400 字符
  - 超限时按 score 倒序截断

用法:
  python -m scripts.inject_distilled_to_memory              # 跑一次（静默）
  python -m scripts.inject_distilled_to_memory --verbose    # 显示进度
  python -m scripts.inject_distilled_to_memory --dry-run    # 只看不动
  python -m scripts.inject_distilled_to_memory --index-dir ~/.workbuddy/rag-index
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


# 让 import 找得到 src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 标记区间
AUTO_INJECT_START = "<!-- AUTO_INJECT_START -->"
AUTO_INJECT_END = "<!-- AUTO_INJECT_END -->"

# 字符预算（防御性）
MAX_AUTO_INJECT_CHARS = 2400        # 自动注入区硬上限
MAX_AUTO_INJECT_TARGET = 2200       # 实际目标值（留 200 字符 buffer 给 header 拼接误差）
MAX_PER_PROJECT_CHARS = 500         # 单项目最多占 500 字符（防霸屏）
MAX_CHUNKS_PER_PROJECT = 5          # 单项目最多 5 条
CHUNK_TEXT_TRUNCATE = 80            # 单 chunk 文本截断

# 默认 MEMORY.md 路径（L2 user-level）
DEFAULT_MEMORY_PATH = Path.home() / ".workbuddy" / "MEMORY.md"

# 默认索引目录（与 ingest_wb_memory_oneshot.py 对齐）
DEFAULT_INDEX_DIR = Path.home() / ".workbuddy" / "rag-index"


def _log(msg: str, verbose: bool = False) -> None:
    """简单日志：verbose 时打印到 stderr"""
    if verbose:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def _truncate(text: str, n: int = CHUNK_TEXT_TRUNCATE) -> str:
    """截断 chunk 文本，统一省略号

    处理：
    - 真实换行 → 空格
    - 字面转义符（\\n / \\r / \\t）→ 空格
    - 多余空白 → 单空格
    """
    import re
    # 把字面转义符（反斜杠 + n/r/t）替换成空格
    text = re.sub(r"\\[nrt]", " ", text)
    # 把真实换行/制表符替换成空格
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # 合并多余空白
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > n:
        return text[:n].rstrip() + "…"
    return text


def _build_inject_block(groups: dict, total_chunks: int) -> str:
    """根据 distill 结果构造自动注入 markdown 块

    Args:
        groups: distill 输出的 {project: {kept_top: [...], kept_count: N, ...}}
        total_chunks: 总 chunks 数（用于显示）

    Returns:
        完整的 markdown 字符串（不含标记）
    """
    if not groups:
        return f"_（当前 RAG 索引共 {total_chunks} chunks，无项目可蒸馏）_"

    # 按项目总分（Σscore）倒序排，热门项目优先
    proj_ranking = sorted(
        groups.items(),
        key=lambda kv: sum(c["score"] for c in kv[1].get("kept_top", [])),
        reverse=True,
    )

    lines = []
    header = (f"_Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
              f"{len(groups)} projects · {total_chunks} chunks total_")
    lines.append(header)
    lines.append("")

    chars_used = len(header) + 1  # +1 for newline

    for proj, info in proj_ranking:
        kept_top = info.get("kept_top", [])
        if not kept_top:
            continue
        # 单项目最多 MAX_CHUNKS_PER_PROJECT 条
        kept_top = kept_top[:MAX_CHUNKS_PER_PROJECT]

        proj_chars_this_round = 0
        proj_header = f"### {proj} ({len(kept_top)} chunks)"
        # 双重检查：全局预算 + 单项目预算
        if chars_used + len(proj_header) + 1 > MAX_AUTO_INJECT_TARGET:
            break
        lines.append(proj_header)
        chars_used += len(proj_header) + 1
        proj_chars_this_round += len(proj_header) + 1

        for c in kept_top:
            score = c.get("score", 0.0)
            access = c.get("access", 0)
            text = _truncate(c.get("text", ""))
            line = f"- `{score:.3f}` acc={access}  {text}"
            line_cost = len(line) + 1

            # 字符预算三道关卡
            if chars_used + line_cost > MAX_AUTO_INJECT_TARGET:
                # 全局预算满 → 后续都截断
                if not any("全局字符预算满" in l for l in lines):
                    lines.append("- _…（全局字符预算满）_")
                break
            if proj_chars_this_round + line_cost > MAX_PER_PROJECT_CHARS:
                # 单项目超 500 字符就停这个项目
                break
            lines.append(line)
            chars_used += line_cost
            proj_chars_this_round += line_cost

        if chars_used >= MAX_AUTO_INJECT_TARGET:
            if not any("字符预算截断" in l for l in lines):
                lines.append("")
                lines.append("_…（后续项目因字符预算截断）_")
            break

        lines.append("")

    return "\n".join(lines).rstrip()


def _read_memory_split(memory_path: Path) -> tuple[str, str]:
    """读 MEMORY.md，分成（手写区, 自动注入区内容）

    Returns:
        (manual_block, auto_block)：
        - manual_block: 标记区间之外的原文（手写部分 + 前后空行）
        - auto_block: 标记区间之内的内容（可能为空字符串）
    """
    if not memory_path.exists():
        return "", ""

    text = memory_path.read_text(encoding="utf-8")

    start_idx = text.find(AUTO_INJECT_START)
    end_idx = text.find(AUTO_INJECT_END)

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        # 没有标记区间 → 返回空 before/after，让 _assemble_final 走"追加"模式
        return "", ""

    # 标记区间在文件中的位置
    # 包含 AUTO_INJECT_START 之前 + AUTO_INJECT_END 之后
    # 重写时只替换两个标记之间的内容
    before = text[: start_idx + len(AUTO_INJECT_START)]
    after = text[end_idx:]

    auto_content = text[start_idx + len(AUTO_INJECT_START) : end_idx].strip("\n")

    # 重组：before + auto_content + after
    # 但我们要返回的是 (manual_block, auto_block) 给上层处理
    # manual_block = 标记区间外的"手写部分"（含标记）
    # auto_block = 标记区间内的当前内容

    # 为简化：返回"前段（含 START 标记）"和"后段（含 END 标记）"
    # 上层重组：manual_before + NEW_BODY + manual_after
    return before, after


def _assemble_final(
    manual_before: str,
    manual_after: str,
    new_auto_body: str,
    memory_path: Path = DEFAULT_MEMORY_PATH,
) -> str:
    """拼装最终 MEMORY.md 内容

    Args:
        manual_before: START 标记之前的部分（手写 + 标记本身）
        manual_after: END 标记之后的部分
        new_auto_body: 新的自动注入内容（不含标记）
        memory_path: MEMORY.md 路径（默认 L2 user-level 路径）。
            只在"无标记 → 追加到文件末尾"分支用到——必须传入以遵守
            --memory-path 覆盖（之前误读 DEFAULT_MEMORY_PATH 全局常量
            会让 --memory-path /tmp/x.md 在无标记场景下污染 ~/.workbuddy/MEMORY.md）。

    Returns:
        完整文件内容
    """
    if not manual_before and not manual_after:
        # 文件本来没有标记 → 整段加到末尾，必须带标记让下次能正确替换
        body = (
            f"\n\n{AUTO_INJECT_START}\n"
            f"## 🤖 自动蒸馏骨架\n\n"
            f"{new_auto_body}\n"
            f"{AUTO_INJECT_END}\n"
        )
        # 如果文件不存在，加个最小头部
        if not memory_path.exists():
            return f"# WorkBuddy 长期记忆\n{body}"
        # 文件存在但没标记 → 追加到末尾
        existing = memory_path.read_text(encoding="utf-8")
        return existing.rstrip() + "\n" + body

    # 有标记区间 → 替换两个标记之间
    return f"{manual_before}\n{new_auto_body}\n{manual_after}"


def run_inject(
    index_dir: Path = DEFAULT_INDEX_DIR,
    memory_path: Path = DEFAULT_MEMORY_PATH,
    verbose: bool = False,
    dry_run: bool = False,
) -> dict:
    """主入口：跑一次 inject

    Returns:
        {
            "ok": bool,
            "before_chars": int,        # 注入前 MEMORY.md 字符数
            "after_chars": int,         # 注入后字符数
            "auto_block_chars": int,    # 自动注入区字符数
            "projects_injected": int,   # 注入了几个项目
            "chunks_injected": int,     # 注入了多少 chunk 条目
            "skipped": bool,            # 是否跳过（无变化）
        }
    """
    try:
        from src.memory import Memory
        from scripts.distill import distill
    except ImportError as e:
        return {"ok": False, "error": f"import failed: {e}"}

    # 1. 蒸馏
    _log(f"开始蒸馏: index_dir={index_dir}", verbose)
    mem = Memory(index_dir=str(index_dir))
    report = distill(mem, top_per_group=MAX_CHUNKS_PER_PROJECT, dry_run=True)
    groups = report.get("groups", {})
    total_chunks = report.get("summary", {}).get("total_chunks", 0)
    _log(f"蒸馏完成: {len(groups)} 项目 / {total_chunks} chunks", verbose)

    # 2. 拼装注入块
    new_auto_body = _build_inject_block(groups, total_chunks)
    _log(f"新自动注入块: {len(new_auto_body)} 字符", verbose)

    # 3. 读 MEMORY.md 现状
    before_chars = memory_path.read_text(encoding="utf-8").__len__() if memory_path.exists() else 0
    manual_before, manual_after = _read_memory_split(memory_path)

    # 4. 拼装
    final_text = _assemble_final(manual_before, manual_after, new_auto_body, memory_path)
    after_chars = len(final_text)

    # 5. 统计实际注入的项目和 chunks
    projects_injected = sum(1 for line in new_auto_body.split("\n") if line.startswith("### "))
    chunks_injected = sum(1 for line in new_auto_body.split("\n") if line.startswith("- `"))

    # 6. dry-run 不写
    if dry_run:
        _log(f"[DRY-RUN] 将写入 {after_chars} 字符 (原 {before_chars})", verbose)
        return {
            "ok": True,
            "dry_run": True,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "auto_block_chars": len(new_auto_body),
            "projects_injected": projects_injected,
            "chunks_injected": chunks_injected,
        }

    # 7. 实际写入
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(final_text, encoding="utf-8")
    _log(f"✅ 已写入: {memory_path} ({after_chars} 字符)", verbose)

    return {
        "ok": True,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "auto_block_chars": len(new_auto_body),
        "projects_injected": projects_injected,
        "chunks_injected": chunks_injected,
    }


def main():
    parser = argparse.ArgumentParser(description="D 方案: 蒸馏 → 注入到 MEMORY.md")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR),
                        help=f"RAG 索引目录（默认 {DEFAULT_INDEX_DIR}）")
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY_PATH),
                        help=f"MEMORY.md 路径（默认 {DEFAULT_MEMORY_PATH}）")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示进度")
    parser.add_argument("--dry-run", action="store_true", help="只看不写")
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args()

    result = run_inject(
        index_dir=Path(args.index_dir),
        memory_path=Path(args.memory_path),
        verbose=args.verbose,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if not result.get("ok"):
            print(f"❌ inject 失败: {result.get('error', '?')}", file=sys.stderr)
            sys.exit(1)
        if result.get("dry_run"):
            print(f"[DRY-RUN] 注入块 {result['auto_block_chars']} 字符，"
                  f"包含 {result['projects_injected']} 项目 / {result['chunks_injected']} chunks")
        else:
            print(f"✅ MEMORY.md 已更新: {result['before_chars']} → {result['after_chars']} 字符")
            print(f"   自动注入区: {result['auto_block_chars']} 字符，"
                  f"{result['projects_injected']} 项目 / {result['chunks_injected']} chunks")


if __name__ == "__main__":
    sys.exit(main() or 0)
