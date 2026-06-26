"""
scripts.health 回归测试

重点：索引为空时 _check_index 不能抛 ZeroDivisionError
（regression: 2026-06-20 production index bug）
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.health import _check_index


class FakeIndexer:
    """模拟 indexer：可控 total / all_chunks"""

    def __init__(self, total: int, chunks: list):
        self._total = total
        self._chunks = chunks

    def count(self):
        return self._total

    def all_chunks(self):
        return self._chunks


class FakeMemory:
    """只需要 embedder 属性（_check_index 不用）"""
    pass


def test_empty_index_no_division_error():
    """regression: total=0 时不能 ZeroDivisionError"""
    mem = FakeMemory()
    mem.indexer = FakeIndexer(total=0, chunks=[])

    result = _check_index(mem)

    # 不抛异常 + 状态应该是 🟡（空索引）
    assert result["status"] == "🟡", f"expected 🟡, got {result['status']}"
    assert result["total"] == 0
    assert result["avg_length"] == 0
    assert result["avg_confidence"] == 0
    assert result["never_accessed"] == 0
    assert "索引为空" in result["notes"]
    # 关键：no_access / total 的逻辑没有抛异常
    assert not any("90%+" in n for n in result["notes"]), (
        f"total=0 时不该触发死数据警告，实际 notes={result['notes']}"
    )


def test_small_index_no_division_error():
    """边界：total <= 10 时应该跳过死数据检查（即使全未访问）"""
    mem = FakeMemory()
    chunks = [
        {"source": "test", "length": 100, "confidence": 0.5, "access_count": 0}
        for _ in range(5)
    ]
    mem.indexer = FakeIndexer(total=5, chunks=chunks)

    result = _check_index(mem)

    # 5 < 10，不应该触发"90%+"提示
    assert not any("90%+" in n for n in result["notes"]), (
        f"total=5 < 10 时不该触发死数据警告，实际 notes={result['notes']}"
    )


def test_large_index_triggers_dead_data_warning():
    """total > 10 且 90%+ 未访问 → 应该触发死数据警告"""
    mem = FakeMemory()
    chunks = [
        {"source": "test", "length": 100, "confidence": 0.5, "access_count": 0}
        for _ in range(20)  # 全部 access_count=0
    ]
    mem.indexer = FakeIndexer(total=20, chunks=chunks)

    result = _check_index(mem)

    assert any("90%+" in n for n in result["notes"]), (
        f"total=20 全未访问时该触发死数据警告，实际 notes={result['notes']}"
    )


def test_mixed_index_no_warning():
    """total=20, 5 个已访问 → 25% 未访问，不该触发"""
    mem = FakeMemory()
    chunks = []
    for i in range(20):
        chunks.append({
            "source": "test",
            "length": 100,
            "confidence": 0.5,
            "access_count": 1 if i < 5 else 0,
        })
    mem.indexer = FakeIndexer(total=20, chunks=chunks)

    result = _check_index(mem)

    # 15/20 = 75% 未访问 < 90%
    assert not any("90%+" in n for n in result["notes"]), (
        f"75% 未访问不该触发警告，实际 notes={result['notes']}"
    )


if __name__ == "__main__":
    test_empty_index_no_division_error()
    print("✓ test_empty_index_no_division_error")
    test_small_index_no_division_error()
    print("✓ test_small_index_no_division_error")
    test_large_index_triggers_dead_data_warning()
    print("✓ test_large_index_triggers_dead_data_warning")
    test_mixed_index_no_warning()
    print("✓ test_mixed_index_no_warning")
    print("\n✅ 全部 4 个用例通过")
