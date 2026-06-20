"""
注册 ingest_wb_memory_oneshot.py 到 Windows 用户登录 Run 键。

为什么用 HKCU\\...\\Run 而不是 Task Scheduler？
- Run 键在用户登录后立即触发，几乎与 WorkBuddy 启动同步
- 不需要管理员权限（Task Scheduler 的某些模式需要）
- 用户卸载 WorkBuddy 时，自动失效（Run 键值还在但命令找不到）
- 静默执行，没有弹窗

Run 键值结构：
  键: HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
  名: WorkBuddy-RAG-Bootstrap
  值: "C:\\path\\to\\venv\\python.exe" "C:\\path\\to\\oneshot.py"

Linux: 写到 ~/.config/autostart/workbuddy-rag-bootstrap.desktop
macOS: 写到 ~/Library/LaunchAgents/...plist

用法:
  python install_bootstrap.py                # 安装（Windows 自动 / Linux 写 desktop）
  python install_bootstrap.py --uninstall    # 移除
  python install_bootstrap.py --run-now      # 立即跑一次（验证）
"""
from __future__ import annotations
import argparse
import os
import platform
import subprocess
import sys
import winreg  # Windows only
from pathlib import Path


TASK_NAME = "WorkBuddy-RAG-Bootstrap"
SCRIPT_PATH = Path(__file__).resolve().parent / "ingest_wb_memory_oneshot.py"
LOG_PATH = Path.home() / ".workbuddy" / "rag-bootstrap.log"


def _python_exe() -> str:
    """oneshot 用的 python 解释器（优先 venv）"""
    project_root = Path(__file__).resolve().parent.parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    managed = r"C:\Users\JJ\.workbuddy\binaries\python\versions\3.13.12\python.exe"
    if os.path.exists(managed):
        return managed
    return sys.executable


def install_windows() -> tuple[bool, str]:
    """注册到 HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run"""
    python_exe = _python_exe()
    cmd = f'"{python_exe}" "{SCRIPT_PATH}"'
    # 额外加 PYTHONIOENCODING=utf-8 + HF_HUB_OFFLINE=1
    cmd = f'set PYTHONIOENCODING=utf-8&& set HF_HUB_OFFLINE=1&& "{python_exe}" "{SCRIPT_PATH}"'

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE,
        )
        winreg.SetValueEx(key, TASK_NAME, 0, winreg.REG_SZ, cmd)
        winreg.CloseKey(key)

        # 验证
        verify_key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_READ,
        )
        actual, _ = winreg.QueryValueEx(verify_key, TASK_NAME)
        winreg.CloseKey(verify_key)

        return True, (
            f"✅ 已注册 Run 键值:\n"
            f"  HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\{TASK_NAME}\n"
            f"  = {actual}\n"
            f"  日志: {LOG_PATH}\n"
            f"  下次 Windows 登录后自动触发"
        )
    except OSError as e:
        return False, f"注册表写入失败: {e}"
    except Exception as e:
        return False, f"未知错误: {e}"


def uninstall_windows() -> tuple[bool, str]:
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE,
        )
        try:
            winreg.DeleteValue(key, TASK_NAME)
            return True, f"✅ 已删除 Run 键值: {TASK_NAME}"
        except FileNotFoundError:
            return True, "Run 键值本来就不存在（无害）"
        finally:
            winreg.CloseKey(key)
    except OSError as e:
        return False, f"删除失败: {e}"


def install_linux() -> tuple[bool, str]:
    """写 XDG autostart desktop entry"""
    python_exe = _python_exe()
    autostart_dir = Path.home() / ".config" / "autostart"
    autostart_dir.mkdir(parents=True, exist_ok=True)
    desktop_file = autostart_dir / f"{TASK_NAME.lower()}.desktop"

    content = f"""[Desktop Entry]
Type=Application
Name=WorkBuddy RAG Bootstrap
Comment=WorkBuddy 启动时扫描 ~/.workbuddy/memory/ 入库
Exec=env PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 "{python_exe}" "{SCRIPT_PATH}"
Terminal=false
Hidden=false
X-GNOME-Autostart-enabled=true
"""
    desktop_file.write_text(content, encoding="utf-8")
    return True, f"✅ 已写入: {desktop_file}"


def uninstall_linux() -> tuple[bool, str]:
    desktop_file = Path.home() / ".config" / "autostart" / f"{TASK_NAME.lower()}.desktop"
    if desktop_file.exists():
        desktop_file.unlink()
        return True, f"✅ 已删除: {desktop_file}"
    return True, "desktop entry 本来就不存在（无害）"


def run_now() -> tuple[bool, str]:
    """立即跑一次"""
    python_exe = _python_exe()
    cmd = [python_exe, str(SCRIPT_PATH), "--verbose"]
    print(f"🚀 立即执行: {' '.join(cmd)}")
    print("=" * 70)
    try:
        r = subprocess.run(
            cmd,
            cwd=str(SCRIPT_PATH.parent.parent),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"},
            timeout=300,
        )
        return r.returncode == 0, f"退出码: {r.returncode}\n日志: {LOG_PATH}"
    except subprocess.TimeoutExpired:
        return False, "oneshot 超时（>5 分钟）"


def main():
    parser = argparse.ArgumentParser(description="WorkBuddy 启动钩子安装器")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--run-now", action="store_true")
    args = parser.parse_args()

    system = platform.system()

    if args.uninstall:
        if system == "Windows":
            ok, msg = uninstall_windows()
        elif system == "Linux":
            ok, msg = uninstall_linux()
        else:
            ok, msg = False, f"❌ {system} 暂不支持自动安装"
        print(msg)
        return 0 if ok else 1

    if args.run_now:
        ok, msg = run_now()
        print(msg)
        return 0 if ok else 1

    # 默认安装
    if system == "Windows":
        ok, msg = install_windows()
    elif system == "Linux":
        ok, msg = install_linux()
    else:
        ok, msg = False, f"❌ {system} 暂不支持自动安装"

    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())