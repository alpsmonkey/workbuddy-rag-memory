"""
测试 bootstrap 链路：
1. oneshot 脚本能跑（写入日志 + 返回 0/1）
2. install_bootstrap.py 注册表读写正常
3. scan_workbuddy_memory 能识别 YYYY-MM-DD.md 文件

这些测试不依赖 bge-m3 模型（用 mock embedder）
"""
from __future__ import annotations
import json
import os
import sys
import tempfile
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import Memory, DEFAULT_WB_MEMORY_PATTERNS


def test_scan_includes_yyyy_mm_dd():
    """核心修复：YYYY-MM-DD.md 应该被默认 glob 覆盖"""
    assert any(r"\d{4}" in p or "[0-9]" in p for p in DEFAULT_WB_MEMORY_PATTERNS), \
        f"DEFAULT_WB_MEMORY_PATTERNS 应包含 YYYY-MM-DD 模式，实际: {DEFAULT_WB_MEMORY_PATTERNS}"

    # 端到端：写一个临时 memory 文件 + 跑 scan，看能否找到
    with tempfile.TemporaryDirectory() as tmp:
        mem_dir = Path(tmp) / ".workbuddy" / "memory"
        mem_dir.mkdir(parents=True)
        test_file = mem_dir / "2026-06-20.md"
        test_file.write_text("# 测试\nSkillFather 是 Python 项目。\n", encoding="utf-8")

        # 用 mock embedder（避免 bge-m3 加载）
        class MockEmbedder:
            dim = 4
            def embed(self, text):
                import numpy as np
                np.random.seed(hash(text) % 100)
                arr = np.random.rand(self.dim)
                return arr  # numpy array，dedup 会调 .tolist()

        mem = Memory(index_dir=tmp + "/.index", embedder=MockEmbedder())
        result = mem.scan_workbuddy_memory(dirs=[str(mem_dir.parent)])

        assert result["scanned"] >= 1, f"应扫到 1 个文件，实际: {result}"
        assert any("2026-06-20" in f["path"] for f in result["files"]), \
            f"应扫到 2026-06-20.md，实际文件: {result['files']}"
        print("✅ YYYY-MM-DD.md 被默认 glob 覆盖")


def test_oneshot_writes_log():
    """oneshot 跑完必须写日志文件"""
    import subprocess
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("⏭️  跳过（venv 不存在）")
        return

    log_path = Path.home() / ".workbuddy" / "rag-bootstrap.log"

    # 跑前清空日志
    if log_path.exists():
        log_path.unlink()

    r = subprocess.run(
        [str(venv_python), str(PROJECT_ROOT / "scripts" / "ingest_wb_memory_oneshot.py")],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=180,
    )
    assert r.returncode in (0, 1), f"oneshot 返回码异常: {r.returncode}"
    assert log_path.exists(), f"日志文件未生成: {log_path}"

    content = log_path.read_text(encoding="utf-8")
    assert "启动 ingest_wb_memory_oneshot" in content
    assert "完成" in content
    print(f"✅ oneshot 写日志成功（{log_path}, {len(content)} 字节）")


def test_install_bootstrap_dry_run():
    """install_bootstrap.py 不实际改注册表时也能解析参数"""
    import subprocess
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("⏭️  跳过（venv 不存在）")
        return
    if platform.system() != "Windows":
        print("⏭️  跳过（非 Windows）")
        return

    # --help 必须能跑
    r = subprocess.run(
        [str(venv_python), str(PROJECT_ROOT / "scripts" / "install_bootstrap.py"), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    assert "WorkBuddy" in r.stdout or "bootstrap" in r.stdout.lower()
    print("✅ install_bootstrap.py --help 正常")


def test_install_bootstrap_registry_roundtrip():
    """注册表读写完整回路（实际写入 + 读回 + 删除）"""
    if platform.system() != "Windows":
        print("⏭️  跳过（非 Windows）")
        return

    import winreg

    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("⏭️  跳过（venv 不存在）")
        return

    import subprocess
    script = PROJECT_ROOT / "scripts" / "install_bootstrap.py"

    # 1. 安装
    r = subprocess.run(
        [str(venv_python), str(script)],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"安装失败: {r.stderr}"
    assert "WorkBuddy-RAG-Bootstrap" in r.stdout

    # 2. 读注册表验证
    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0, winreg.KEY_READ,
    )
    value, _ = winreg.QueryValueEx(key, "WorkBuddy-RAG-Bootstrap")
    winreg.CloseKey(key)
    assert "ingest_wb_memory_oneshot" in value
    assert "python.exe" in value

    # 3. 卸载
    r2 = subprocess.run(
        [str(venv_python), str(script), "--uninstall"],
        capture_output=True, text=True, timeout=30,
    )
    assert r2.returncode == 0

    # 4. 验证已删
    key2 = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        0, winreg.KEY_READ,
    )
    try:
        winreg.QueryValueEx(key2, "WorkBuddy-RAG-Bootstrap")
        winreg.CloseKey(key2)
        assert False, "键值应该被删除"
    except FileNotFoundError:
        winreg.CloseKey(key2)
        print("✅ 注册表 4 步回路通过（install → query → uninstall → verify gone）")


def test_skill_bridge_call():
    """rag_bootstrap skill main.py 能跑"""
    skill_main = Path(r"C:\Users\JJ\.workbuddy\skills\rag_bootstrap\main.py")
    if not skill_main.exists():
        print(f"⏭️  跳过（{skill_main} 不存在）")
        return

    import subprocess
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print("⏭️  跳过（venv 不存在）")
        return

    r = subprocess.run(
        [str(venv_python), str(skill_main)],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
        capture_output=True, text=True, timeout=180,
    )
    assert r.returncode in (0, 1)
    assert "日志:" in r.stdout
    print(f"✅ rag_bootstrap skill 跑通（exit={r.returncode}）")


def main():
    tests = [
        test_scan_includes_yyyy_mm_dd,
        test_oneshot_writes_log,
        test_install_bootstrap_dry_run,
        test_install_bootstrap_registry_roundtrip,
        test_skill_bridge_call,
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
    print(f"  bootstrap: {passed} passed, {failed} failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())