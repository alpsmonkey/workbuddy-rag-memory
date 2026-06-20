"""
tests/test_inject_distilled_to_memory.py

D 方案 inject 单元测试：
  1. 字符截断（80 字符 + 省略号）
  2. 标记区间识别（START/END 标记）
  3. 拼装逻辑（有标记替换 vs 无标记追加）
  4. 字符预算（三道关卡）
  5. 控制字符清理（\\n \\r \\t + 字面转义）
  6. 集成：distill() 输出格式兼容
"""
from __future__ import annotations
import sys
from pathlib import Path

# 让 import 找得到 scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inject_distilled_to_memory import (
    _truncate,
    _read_memory_split,
    _assemble_final,
    _build_inject_block,
    AUTO_INJECT_START,
    AUTO_INJECT_END,
    MAX_AUTO_INJECT_TARGET,
    MAX_PER_PROJECT_CHARS,
    MAX_CHUNKS_PER_PROJECT,
)


# ========================================================================
# 1. _truncate
# ========================================================================

def test_truncate_short_text():
    """短文本不截断"""
    assert _truncate("hello world") == "hello world"


def test_truncate_long_text():
    """长文本截断到 80 字符 + 省略号"""
    long_text = "a" * 200
    result = _truncate(long_text)
    assert result.endswith("…")
    assert len(result) == 81  # 80 + 省略号
    assert result.startswith("a" * 80)


def test_truncate_newlines_to_spaces():
    """真实换行替换成空格"""
    assert _truncate("line1\nline2\nline3") == "line1 line2 line3"


def test_truncate_literal_escape_sequences():
    """字面转义符（反斜杠 + n/r/t）替换成空格"""
    # \\n 是字面 2 字符（反斜杠 + n）
    assert _truncate(r"foo\nbar") == "foo bar"
    assert _truncate(r"foo\tbar") == "foo bar"
    assert _truncate(r"foo\rbar") == "foo bar"


def test_truncate_combined():
    """混合：字面 \\n + 真实换行 + 多余空白"""
    text = r"line1\n\nline2  \n  line3"
    result = _truncate(text)
    # \\n → 空格, 真实 \n → 空格, 多余空白 → 单空格
    assert result == "line1 line2 line3"


# ========================================================================
# 2. _assemble_final
# ========================================================================

def test_assemble_no_markers_append_mode():
    """无标记 → 追加模式（带 START/END 标记）"""
    result = _assemble_final("", "", "new content")
    assert AUTO_INJECT_START in result
    assert AUTO_INJECT_END in result
    assert "new content" in result


def test_assemble_with_markers_replace():
    """有标记 → 替换两个标记之间（before/after 是 _read_memory_split 实际输出）"""
    # 模拟 _read_memory_split 的真实输出：before 只到 START 标记末尾
    before = f"manual\n\n{AUTO_INJECT_START}"
    after = f"{AUTO_INJECT_END}\nmanual after"
    result = _assemble_final(before, after, "NEW auto body")
    assert "manual\n\n" in result
    assert "NEW auto body" in result
    # before 不含标记间内容，所以"old auto"不在 result 中
    assert "old auto" not in result
    assert "manual after" in result


def test_assemble_replace_does_not_duplicate_manual():
    """替换模式不会重复手写部分"""
    manual = "user-written 偏好"
    # before 模拟 _read_memory_split 输出：只到 START 末尾
    before = f"{manual}\n\n{AUTO_INJECT_START}"
    after = f"{AUTO_INJECT_END}\n"
    result = _assemble_final(before, after, "fresh")
    # manual 只出现 1 次
    assert result.count(manual) == 1
    # fresh 在位
    assert "fresh" in result
    # START/END 各 1 次
    assert result.count(AUTO_INJECT_START) == 1
    assert result.count(AUTO_INJECT_END) == 1


# ========================================================================
# 3. _build_inject_block
# ========================================================================

def _make_chunk(score: float, text: str, access: int = 0) -> dict:
    return {"score": score, "text": text, "access": access, "ts": "2026-01-01", "project": "test"}


def test_build_inject_block_empty_groups():
    """空 groups → 显示提示"""
    block = _build_inject_block({}, 0)
    assert "无项目可蒸馏" in block


def test_build_inject_block_basic():
    """基本功能：1 个项目 + 2 chunks"""
    groups = {
        "TestProj": {
            "kept_top": [
                _make_chunk(0.9, "first chunk text", 10),
                _make_chunk(0.7, "second chunk text", 5),
            ]
        }
    }
    block = _build_inject_block(groups, 100)
    assert "TestProj" in block
    assert "0.900" in block
    assert "0.700" in block
    assert "first chunk text" in block
    assert "second chunk text" in block
    assert "100 chunks" in block


def test_build_inject_block_projects_sorted_by_total_score():
    """项目按总分倒序排"""
    groups = {
        "LowProj": {
            "kept_top": [_make_chunk(0.1, "low", 0)]
        },
        "HighProj": {
            "kept_top": [_make_chunk(0.9, "high1", 100), _make_chunk(0.8, "high2", 100)]
        },
    }
    block = _build_inject_block(groups, 10)
    # HighProj 应在 LowProj 之前
    high_pos = block.find("HighProj")
    low_pos = block.find("LowProj")
    assert high_pos < low_pos
    assert high_pos > 0


def test_build_inject_block_respects_max_chunks_per_project():
    """单项目最多 5 条"""
    groups = {
        "BigProj": {
            "kept_top": [_make_chunk(0.9 - i * 0.01, f"chunk-{i}", i) for i in range(10)]
        }
    }
    block = _build_inject_block(groups, 10)
    # 只显示前 5 条
    for i in range(5):
        assert f"chunk-{i}" in block
    # 6 之后被截
    assert "chunk-5" not in block


def test_build_inject_block_respects_global_budget():
    """全局字符预算硬上限"""
    # 制造 20 个项目，每个 5 条长文本
    groups = {
        f"Proj{i:02d}": {
            "kept_top": [_make_chunk(0.9, "x" * 100, 10) for _ in range(5)]
        }
        for i in range(20)
    }
    block = _build_inject_block(groups, 1000)
    # 块长 < 目标值 + 单行长度（容差）
    assert len(block) <= MAX_AUTO_INJECT_TARGET + 200


def test_build_inject_block_truncates_long_text():
    """长 chunk 文本截断到 80 字符"""
    long_text = "y" * 200
    groups = {
        "TestProj": {
            "kept_top": [_make_chunk(0.9, long_text, 0)]
        }
    }
    block = _build_inject_block(groups, 1)
    # y 出现不超过 80 次 + 省略号
    assert "y" * 80 + "…" in block


# ========================================================================
# 4. _read_memory_split
# ========================================================================

def test_read_split_no_markers(tmp_path):
    """无标记 → 返回 ("", "")"""
    p = tmp_path / "MEMORY.md"
    p.write_text("user manual content\n", encoding="utf-8")
    before, after = _read_memory_split(p)
    assert before == ""
    assert after == ""


def test_read_split_with_markers(tmp_path):
    """有标记 → 标记边界正确"""
    p = tmp_path / "MEMORY.md"
    text = f"user\n\n{AUTO_INJECT_START}\nauto content\n{AUTO_INJECT_END}\nend"
    p.write_text(text, encoding="utf-8")
    before, after = _read_memory_split(p)
    # before 应包含 START 标记
    assert AUTO_INJECT_START in before
    assert "user" in before
    # after 应包含 END 标记
    assert AUTO_INJECT_END in after
    assert "end" in after


def test_read_split_nonexistent(tmp_path):
    """文件不存在 → 返回 ("", "")"""
    p = tmp_path / "MEMORY.md"
    before, after = _read_memory_split(p)
    assert before == ""
    assert after == ""


# ========================================================================
# 5. 集成：完整流程幂等性（用临时文件）
# ========================================================================

def test_full_round_trip_idempotent(tmp_path, monkeypatch):
    """完整流程：第一次写 → 第二次替换 → 第三次稳定"""
    import scripts.inject_distilled_to_memory as m

    # 重定向 DEFAULT_MEMORY_PATH 到临时文件
    fake_mem = tmp_path / "MEMORY.md"
    fake_mem.write_text("# 跨项目通用记忆\n\n# 鹏哥偏好: 测试\n", encoding="utf-8")
    monkeypatch.setattr(m, "DEFAULT_MEMORY_PATH", fake_mem)

    # mock distill 避免依赖 bge-m3
    def fake_distill(mem, top_per_group=5, dry_run=False):
        return {
            "groups": {
                "TestProj": {
                    "kept_top": [
                        {"score": 0.9, "text": "chunk-1", "access": 10, "ts": "2026-01-01", "project": "TestProj"},
                        {"score": 0.7, "text": "chunk-2", "access": 5, "ts": "2026-01-01", "project": "TestProj"},
                    ],
                    "kept_count": 2,
                    "total": 2,
                    "purge_count": 0,
                    "purge_candidates": [],
                }
            },
            "summary": {"total_chunks": 100, "projects": 1, "purged": 0,
                        "purge_threshold": 0.005, "avg_score": 0.8, "dry_run": False},
        }

    import scripts.distill as d
    monkeypatch.setattr(d, "distill", fake_distill)

    # mock src.memory.Memory（inject 函数内部 from src.memory import Memory）
    import src.memory as sm
    class FakeMem:
        def __init__(self, index_dir):
            pass
    monkeypatch.setattr(sm, "Memory", FakeMem)

    # 第一次
    r1 = m.run_inject(memory_path=fake_mem, verbose=False, dry_run=False)
    assert r1["ok"]
    first_text = fake_mem.read_text(encoding="utf-8")
    assert AUTO_INJECT_START in first_text
    assert AUTO_INJECT_END in first_text
    assert "TestProj" in first_text
    assert "# 鹏哥偏好: 测试" in first_text  # 手写部分不动

    # 第二次
    r2 = m.run_inject(memory_path=fake_mem, verbose=False, dry_run=False)
    second_text = fake_mem.read_text(encoding="utf-8")
    # 第二次不应有重复内容
    assert second_text.count(AUTO_INJECT_START) == 1
    assert second_text.count(AUTO_INJECT_END) == 1
    # 手写部分还是只出现 1 次
    assert second_text.count("# 鹏哥偏好: 测试") == 1

    # 第三次
    r3 = m.run_inject(memory_path=fake_mem, verbose=False, dry_run=False)
    third_text = fake_mem.read_text(encoding="utf-8")
    assert third_text.count(AUTO_INJECT_START) == 1
    assert third_text.count(AUTO_INJECT_END) == 1
    # 字符数稳定（除了时间戳变化）
    # 移除时间戳行后比较
    import re
    stable = re.sub(r"_Last update:.*?_", "_Last update: stable_", third_text)
    assert re.sub(r"_Last update:.*?_", "_Last update: stable_", second_text) == stable


# ========================================================================
# 6. 关键不变量：手写部分永不被覆盖
# ========================================================================

def test_manual_part_never_overwritten(tmp_path, monkeypatch):
    """用户手写部分（标记外）永不被覆盖"""
    import scripts.inject_distilled_to_memory as m

    fake_mem = tmp_path / "MEMORY.md"
    manual = "# 鹏哥偏好: 严格测试用 \n# 不能动: 这部分\n"
    fake_mem.write_text(manual, encoding="utf-8")
    monkeypatch.setattr(m, "DEFAULT_MEMORY_PATH", fake_mem)

    def fake_distill(mem, top_per_group=5, dry_run=False):
        return {
            "groups": {
                "AnyProj": {
                    "kept_top": [{"score": 0.5, "text": "test", "access": 0,
                                  "ts": "2026-01-01", "project": "AnyProj"}],
                }
            },
            "summary": {"total_chunks": 1, "projects": 1, "purged": 0,
                        "purge_threshold": 0.005, "avg_score": 0.5, "dry_run": False},
        }

    import scripts.distill as d
    monkeypatch.setattr(d, "distill", fake_distill)

    import src.memory as sm
    class FakeMem:
        def __init__(self, index_dir):
            pass
    monkeypatch.setattr(sm, "Memory", FakeMem)

    m.run_inject(memory_path=fake_mem, verbose=False, dry_run=False)
    text = fake_mem.read_text(encoding="utf-8")
    # 严格断言：手写字符完全保留
    assert "严格测试用" in text
    assert "不能动: 这部分" in text
    # 出现 1 次（不重复）
    assert text.count("严格测试用") == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
