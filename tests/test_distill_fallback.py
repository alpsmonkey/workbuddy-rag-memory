"""
测试 oneshot ingest 的蒸馏兜底逻辑
- 首次从未蒸馏过 → 触发蒸馏
- 距上次 < 24h → 跳过
- 强制改状态文件 last_distill_ts 模拟"过期" → 触发补跑
- 索引 < min_chunks → 跳过
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_first_run_triggers_distill():
    """从未蒸馏过：首次运行应触发"""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from ingest_wb_memory_oneshot import _load_distill_state, _should_distill

    with tempfile.TemporaryDirectory() as tmp:
        idx = Path(tmp)
        state = _load_distill_state(idx)
        assert state["last_distill_ts"] == 0.0

        should, reason = _should_distill(state, current_chunks=100)
        assert should is True
        assert "首次" in reason
        print(f"✅ 首次触发: {reason}")


def test_recent_distill_skipped():
    """距上次 < 24h：跳过"""
    from ingest_wb_memory_oneshot import _should_distill

    state = {"last_distill_ts": time.time() - 3600, "last_total_chunks": 100}  # 1 小时前
    should, reason = _should_distill(state, current_chunks=100)
    assert should is False
    assert "无需补跑" in reason
    print(f"✅ 24h 内跳过: {reason}")


def test_stale_distill_triggers():
    """距上次 > 24h：触发补跑"""
    from ingest_wb_memory_oneshot import _should_distill

    state = {"last_distill_ts": time.time() - 86400 * 2, "last_total_chunks": 100}  # 2 天前
    should, reason = _should_distill(state, current_chunks=100)
    assert should is True
    assert "24h" in reason
    print(f"✅ 过期触发: {reason}")


def test_small_index_skipped():
    """索引 < min_chunks：跳过（避免冷启动浪费）"""
    from ingest_wb_memory_oneshot import _should_distill

    # 即使 last_distill_ts=0（"从未蒸馏"），如果索引小也跳过
    state = {"last_distill_ts": time.time() - 86400 * 7, "last_total_chunks": 20}  # 7 天前 + 索引小
    should, reason = _should_distill(state, current_chunks=20, min_chunks=50)
    # 注意：当前实现下"过期"会优先触发，min_chunks 只在"未过期"时生效
    # 这是一个保守策略——索引小但过期，仍触发（蒸馏是 idempotent 操作）
    # 但如果改成"过期且有内容"才触发，需要 current_chunks 至少 1
    # 当前测试：先验证"过期就触发"行为（索引大小检查不优先于过期）
    assert should is True  # 过期 7 天 > 24h，触发
    print(f"⚠️  小索引+过期: 仍触发（蒸馏 idempotent）：{reason}")

    # 然后测：recent + 小索引 = 跳过
    state_recent = {"last_distill_ts": time.time() - 3600, "last_total_chunks": 20}
    should_recent, reason_recent = _should_distill(state_recent, current_chunks=20, min_chunks=50)
    assert should_recent is False
    assert "20 chunks" in reason_recent
    print(f"✅ 小索引+recent: 跳过: {reason_recent}")


def test_save_load_distill_state():
    """状态文件读写"""
    from ingest_wb_memory_oneshot import _load_distill_state, _save_distill_state

    with tempfile.TemporaryDirectory() as tmp:
        idx = Path(tmp)
        # 初始：空
        state = _load_distill_state(idx)
        assert state["last_distill_ts"] == 0.0

        # 写
        _save_distill_state(idx, total_chunks=42)
        state = _load_distill_state(idx)
        assert state["last_total_chunks"] == 42
        assert state["last_distill_ts"] > 0

        # 再次保存（更新）
        _save_distill_state(idx, total_chunks=43)
        state = _load_distill_state(idx)
        assert state["last_total_chunks"] == 43
        print(f"✅ 状态文件读写: {state}")


def test_skip_distill_flag():
    """--skip-distill 必须能跳过兜底"""
    # 端到端：跑 ingest_wb_memory_oneshot.py --skip-distill，验证不触发 distill
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("⏭️  跳过（venv 不存在）")
        return

    r = subprocess.run(
        [str(venv_python), str(PROJECT_ROOT / "scripts" / "ingest_wb_memory_oneshot.py"),
         "--skip-distill", "--verbose"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=120,
    )

    # 输出里不应有 "触发蒸馏兜底"
    assert "触发蒸馏兜底" not in r.stdout
    assert "蒸馏完成" not in r.stdout
    print("✅ --skip-distill 跳过兜底（stdout 不含 distill 触发字样）")


def main():
    tests = [
        test_first_run_triggers_distill,
        test_recent_distill_skipped,
        test_stale_distill_triggers,
        test_small_index_skipped,
        test_save_load_distill_state,
        test_skip_distill_flag,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"  distill_fallback: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())