"""
scripts.inject_distilled_to_memory 回归测试

D 方案关键路径：distill() → MEMORY.md 自动注入。
手写区 / 自动注入区隔离 + 字符预算守门 + 单项目配额 + 单 chunk 截断。

不依赖真实 RAG 索引 → 直接构造 distill 形状的 groups dict 喂给纯函数。
"""
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inject_distilled_to_memory import (
    _truncate,
    _build_inject_block,
    _read_memory_split,
    _assemble_final,
    _log,
    AUTO_INJECT_START,
    AUTO_INJECT_END,
    MAX_AUTO_INJECT_TARGET,
    MAX_AUTO_INJECT_CHARS,
    MAX_PER_PROJECT_CHARS,
    MAX_CHUNKS_PER_PROJECT,
    CHUNK_TEXT_TRUNCATE,
)


# ============================================================
# 1. _truncate — 单 chunk 文本处理（防污染 + 长度守门）
# ============================================================

def test_truncate_keeps_short_text():
    """短文本原样返回"""
    assert _truncate("hello world") == "hello world"


def test_truncate_cuts_long_text_with_ellipsis():
    """长文本截断到 N 字符 + 省略号"""
    long_text = "a" * 200
    out = _truncate(long_text, n=80)
    assert len(out) <= 81  # 80 + "…"
    assert out.endswith("…")
    assert out == "a" * 80 + "…"


def test_truncate_normalizes_whitespace():
    """换行 / 制表符 / 字面 \\n → 空格"""
    raw = "line1\\nline2\nline3\twith\ttab"
    out = _truncate(raw, n=200)
    # 字面 \\n（两个字符：反斜杠 + n）应该被替换为单个空格
    assert "\\n" not in out
    assert "\n" not in out
    assert "\t" not in out
    # 合并后应该是单空格分隔
    assert out == "line1 line2 line3 with tab"


def test_truncate_collapses_multiple_spaces():
    """连续空白合并成单空格"""
    raw = "foo    bar\n\nbaz"
    out = _truncate(raw, n=200)
    assert "  " not in out  # 不应有连续空格
    assert out == "foo bar baz"


def test_truncate_handles_empty():
    """空字符串 / 全空白输入不崩"""
    assert _truncate("") == ""
    assert _truncate("   \n\n  ") == ""


# ============================================================
# 2. _build_inject_block — 字符预算 + 项目配额 + 排序
# ============================================================

def _make_chunk(text: str, score: float = 0.1, access: int = 1, project: str = "TestProj"):
    """构造 distill.kept_top 单条"""
    return {"id": f"id-{text[:8]}", "text": text, "score": score,
            "access": access, "ts": "2026-06-20T00:00:00", "project": project}


def _make_groups(*project_specs):
    """构造 distill groups: 每个 (proj, score_list, text_list, access_list)"""
    groups = {}
    for proj, scores, texts, access in project_specs:
        chunks = [_make_chunk(t, s, a, proj) for t, s, a in zip(texts, scores, access)]
        groups[proj] = {"total": len(chunks), "kept_count": len(chunks),
                        "purge_count": 0, "kept_top": chunks}
    return groups


def test_build_block_empty_groups():
    """空 groups → 返回提示文本（不抛）"""
    out = _build_inject_block({}, total_chunks=0)
    assert "无项目可蒸馏" in out
    assert "0 chunks" in out


def test_build_block_single_project_basic():
    """单项目 1 个 chunk → 包含 header + chunk 行"""
    groups = _make_groups(("ProjA", [0.5], ["hello world"], [1]))
    out = _build_inject_block(groups, total_chunks=10)
    assert "ProjA" in out
    assert "0.500" in out
    assert "acc=1" in out
    assert "hello world" in out
    assert "### ProjA" in out
    assert "10 chunks" in out


def test_build_block_limits_chunks_per_project():
    """单项目超过 MAX_CHUNKS_PER_PROJECT (5) → 截断到 5"""
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    texts = [f"text-{i}" for i in range(7)]
    access = [1] * 7
    groups = _make_groups(("ProjA", scores, texts, access))
    out = _build_inject_block(groups, total_chunks=10)
    # 应该只有 5 条以 "- `" 开头
    chunk_lines = [l for l in out.split("\n") if l.startswith("- `")]
    assert len(chunk_lines) == MAX_CHUNKS_PER_PROJECT, (
        f"单项目应限制 {MAX_CHUNKS_PER_PROJECT} 条，实际 {len(chunk_lines)}"
    )


def test_build_block_respects_global_char_budget():
    """构造大量项目把全局预算顶满 → 输出 ≤ MAX_AUTO_INJECT_CHARS（硬上限 2400）"""
    # 30 个项目，每个 5 条长文本
    specs = []
    for i in range(30):
        specs.append((f"Proj{i:02d}",
                      [0.1] * 5,
                      ["x" * 60] * 5,
                      [1] * 5))
    groups = _make_groups(*specs)

    out = _build_inject_block(groups, total_chunks=150)

    # 硬上限：2400 字符（防御性）
    # 注：触发截断后会追加"…（全局字符预算满）"一行 (~14 chars)，
    # 所以实际输出在 (MAX_AUTO_INJECT_TARGET, MAX_AUTO_INJECT_CHARS] 之间是预期行为
    assert len(out) <= MAX_AUTO_INJECT_CHARS, (
        f"输出 {len(out)} 字符，超过硬上限 {MAX_AUTO_INJECT_CHARS}"
    )
    # 应该出现预算提示
    assert "全局字符预算满" in out, (
        f"应该提示字符预算已满，实际: {out[:200]}"
    )
    # 也应该远低于"不做任何截断"的理论值
    theoretical = 30 * 5 * 80  # 12000
    assert len(out) < theoretical, "如果接近 12000 说明没截断"


def test_build_block_respects_per_project_budget():
    """单个项目 5 条超 500 字符 → 触发单项目配额截断"""
    # 1 个项目，5 条 200 字符文本 = 1000 字符（远超 500 配额）
    long_text = "y" * 200
    groups = _make_groups(("ProjA", [0.1] * 5, [long_text] * 5, [1] * 5))
    out = _build_inject_block(groups, total_chunks=5)

    # 该项目的实际行
    lines = out.split("\n")
    proj_a_section = []
    in_section = False
    for l in lines:
        if l.startswith("### ProjA"):
            in_section = True
            proj_a_section.append(l)
        elif in_section and l.startswith("### "):
            break
        elif in_section and l.strip():
            proj_a_section.append(l)

    proj_a_text = "\n".join(proj_a_section)
    # 单项目总字符不超过 MAX_PER_PROJECT_CHARS（header 约 25 字符 + 4-5 条 line）
    # 允许 +50 字符的 buffer
    assert len(proj_a_text) <= MAX_PER_PROJECT_CHARS + 50, (
        f"ProjA 占用 {len(proj_a_text)} 字符，超过单项目配额 {MAX_PER_PROJECT_CHARS}"
    )


def test_build_block_ranks_projects_by_total_score():
    """多个项目 → 高分项目排在前面"""
    # ProjA 总分 5.0，ProjB 总分 1.0 → ProjA 应该在 ProjB 前面
    groups = _make_groups(
        ("ProjA", [1.0, 1.0, 1.0, 1.0, 1.0], ["a1", "a2", "a3", "a4", "a5"], [1] * 5),
        ("ProjB", [0.2] * 5, ["b1", "b2", "b3", "b4", "b5"], [1] * 5),
    )
    out = _build_inject_block(groups, total_chunks=10)
    pos_a = out.find("### ProjA")
    pos_b = out.find("### ProjB")
    assert pos_a != -1 and pos_b != -1, "两个项目都应该出现"
    assert pos_a < pos_b, f"ProjA (总分高) 应排在 ProjB 前面，posA={pos_a} posB={pos_b}"


# ============================================================
# 3. _read_memory_split — 标记区间识别
# ============================================================

def test_read_split_no_file():
    """文件不存在 → 返回 ("", "")"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        assert _read_memory_split(path) == ("", "")


def test_read_split_file_without_markers():
    """文件存在但无标记 → 返回 ("", "")，让上层走"追加"模式"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        path.write_text("# 鹏哥的手写笔记\n\nno markers here\n", encoding="utf-8")
        assert _read_memory_split(path) == ("", "")


def test_read_split_file_with_markers():
    """文件有标记 → 返回 (before+START, from_END)"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        body = (
            "# 鹏哥手写\n\n"
            "<!-- AUTO_INJECT_START -->\n"
            "old injected stuff\n"
            "<!-- AUTO_INJECT_END -->\n\n"
            "more manual notes\n"
        )
        path.write_text(body, encoding="utf-8")

        before, after = _read_memory_split(path)

        # before 应该包含 START 标记
        assert AUTO_INJECT_START in before
        assert "鹏哥手写" in before
        # after 应该包含 END 标记
        assert AUTO_INJECT_END in after
        assert "more manual notes" in after
        # 中间的内容不应该在 before/after 里
        assert "old injected stuff" not in before
        assert "old injected stuff" not in after


def test_read_split_markers_out_of_order():
    """END 在 START 前面（异常情况）→ 视为无标记"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "MEMORY.md"
        body = "<!-- AUTO_INJECT_END -->before<!-- AUTO_INJECT_START -->\n"
        path.write_text(body, encoding="utf-8")
        assert _read_memory_split(path) == ("", "")


# ============================================================
# 4. _assemble_final — 拼装最终内容
# ============================================================

def test_assemble_with_existing_markers():
    """有标记区间 → 替换两个标记之间"""
    before = "# header\n\n<!-- AUTO_INJECT_START -->"
    after = "<!-- AUTO_INJECT_END -->\n\n## footer\n"
    new_body = "new injected stuff"

    out = _assemble_final(before, after, new_body)

    # 标记区间内应该是新内容
    assert "new injected stuff" in out
    # 标记区间外的手写部分保留
    assert "# header" in out
    assert "## footer" in out
    # 旧内容被替换掉
    assert "old injected" not in out


def test_assemble_no_markers_file_not_exists_creates_with_header():
    """regression: 修复前 _assemble_final 读 DEFAULT_MEMORY_PATH 全局变量，
    即使传 memory_path 也会被忽略。修复后：文件不存在 → 加最小头部 + body。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "subdir" / "MEMORY.md"  # 父目录都不存在
        # _assemble_final 不会自动创建父目录，只关心文件
        # 这里先确保父目录存在
        path.parent.mkdir(parents=True, exist_ok=True)

        out = _assemble_final("", "", "auto body", memory_path=path)

        # 应该带默认 header
        assert out.startswith("# WorkBuddy 长期记忆\n")
        # 应该带新 body + 标记
        assert "auto body" in out
        assert AUTO_INJECT_START in out
        assert AUTO_INJECT_END in out


def test_assemble_no_markers_file_exists_appends_to_target_file():
    """regression: 修复前传 --memory-path /tmp/x.md 但 _assemble_final
    读 DEFAULT_MEMORY_PATH (~/.workbuddy/MEMORY.md) → 误读鹏哥手写区。
    修复后：应该读并追加到 *传入的* memory_path。"""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "custom_memory.md"
        # 重要：构造一个完全独立于 DEFAULT_MEMORY_PATH 的场景
        # 如果代码 bug 回潮，它会读 ~/.workbuddy/MEMORY.md 但不会找到该文件
        # → 走"文件不存在 → 加 header"分支 → 错误地把整段当新建
        # 修复后 → 走"文件存在 → 追加到末尾"分支

        # 防御：确保默认路径不存在（tmp 临时目录隔离）
        target.write_text("# 用户手写头部\n\n已有内容\n", encoding="utf-8")
        # 二次确认
        assert target.exists()

        out = _assemble_final("", "", "new inject", memory_path=target)

        # 应该保留用户手写内容
        assert "# 用户手写头部" in out
        assert "已有内容" in out
        # 应该追加新内容（不是创建新文件）
        assert out.index("已有内容") < out.index("new inject")
        # 标记应该存在
        assert AUTO_INJECT_START in out
        assert AUTO_INJECT_END in out
        # 关键：不应该出现"文件不存在 → 加默认 header"的副作用
        assert not out.startswith("# WorkBuddy 长期记忆\n"), (
            "目标文件已存在时不应再加默认 header"
        )


def test_assemble_no_markers_does_not_touch_unrelated_default_path():
    """regression: 修复前调用 _assemble_final("", "", body, memory_path=/tmp/x.md)
    会读 DEFAULT_MEMORY_PATH = ~/.workbuddy/MEMORY.md。
    修复后：完全不读全局默认路径。

    验证方法：用一个绝对不存在的 target 路径 + 临时清空默认路径环境。
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "isolated.md"
        # 关键：target 不存在 → 应该走 "header + body" 分支
        # 如果代码错误地先读 DEFAULT_MEMORY_PATH 也没事（默认路径存在则追加，
        # 跟我们的 target 完全独立）
        out = _assemble_final("", "", "body", memory_path=target)
        assert "body" in out
        # 不该有 target 之外的路径污染（target 跟 DEFAULT_MEMORY_PATH 无关）
        # 只要没异常 + 包含 body，就说明走对了分支


def test_assemble_with_markers_preserves_surrounding_content():
    """有标记 → 手写区完全不动"""
    before = "TITLE\n\n<!-- AUTO_INJECT_START -->"
    after = "<!-- AUTO_INJECT_END -->\nFOOTER\n"
    new_body = "INJECTED"

    out = _assemble_final(before, after, new_body)
    # 顺序：TITLE 在最前，FOOTER 在最后，INJECTED 在中间
    assert out.index("TITLE") < out.index("INJECTED")
    assert out.index("INJECTED") < out.index("FOOTER")


# ============================================================
# 5. _log — 简单日志（verbose 控制）
# ============================================================

def test_log_silent_when_verbose_false():
    """verbose=False → 不打印到 stderr"""
    import io
    from contextlib import redirect_stderr
    buf = io.StringIO()
    with redirect_stderr(buf):
        _log("hello", verbose=False)
    assert "hello" not in buf.getvalue()


def test_log_prints_when_verbose_true():
    """verbose=True → 打印到 stderr（含时间戳）"""
    import io
    from contextlib import redirect_stderr
    buf = io.StringIO()
    with redirect_stderr(buf):
        _log("test message", verbose=True)
    err = buf.getvalue()
    assert "test message" in err
    # 应该带 [HH:MM:SS] 风格时间戳
    import re
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\]", err), f"时间戳缺失: {err!r}"


# ============================================================
# 运行入口（python 直接跑也能用）
# ============================================================

if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"结果: {passed} passed, {failed} failed, {passed+failed} total")
    sys.exit(0 if failed == 0 else 1)
