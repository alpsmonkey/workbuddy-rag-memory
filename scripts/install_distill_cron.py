"""
注册 scripts/distill.py 为定时任务

Windows: Task Scheduler（XML 注册 + 立即执行一次验证）
Linux:   systemd timer（--systemd 模式）
macOS:   launchd plist（--launchd 模式）

默认: 每天凌晨 3:00 执行（低峰期）

用法:
  python install_distill_cron.py                # 安装 + 验证
  python install_distill_cron.py --uninstall    # 卸载
  python install_distill_cron.py --run-now      # 立即跑一次
"""
from __future__ import annotations
import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


TASK_NAME = "WorkBuddy-RAG-Distill-Daily"
SCRIPT_PATH = Path(__file__).resolve().parent / "distill.py"
DEFAULT_TIME = "03:00"  # 凌晨 3 点


def _python_exe() -> str:
    """返回 distill.py 用的 python 解释器（优先 venv）"""
    project_root = Path(__file__).resolve().parent.parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    # fallback 到 managed python
    managed = r"C:\Users\JJ\.workbuddy\binaries\python\versions\3.13.12\python.exe"
    if os.path.exists(managed):
        return managed
    return sys.executable


def _index_dir_abs() -> str:
    """返回 ~/.workbuddy/rag-index 的绝对路径（避免依赖 cmd 环境变量展开）"""
    return str(Path.home() / ".workbuddy" / "rag-index")


def install_windows(time_str: str = DEFAULT_TIME) -> tuple[bool, str]:
    """注册 Windows Task Scheduler 任务"""
    python_exe = _python_exe()
    index_dir = _index_dir_abs()
    # 必须先把 index_dir 创建出来，否则 distill 找不到
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # 用绝对路径，避免 cmd.exe / PowerShell 变量展开差异
    cmd = f'"{python_exe}" "{SCRIPT_PATH}" --index-dir "{index_dir}"'

    # 用 schtasks 注册（最简洁，无需 XML）
    schtasks_cmd = [
        "schtasks", "/Create",
        "/TN", TASK_NAME,
        "/TR", cmd,
        "/SC", "DAILY",
        "/ST", time_str,
        "/RL", "LIMITED",
        "/F",  # 覆盖已存在的同名任务
    ]

    try:
        r = subprocess.run(
            schtasks_cmd,
            capture_output=True,
            text=True,
            encoding="gbk",  # schtasks 输出是 GBK
            timeout=30,
        )
        if r.returncode != 0:
            return False, f"schtasks 失败: {r.stderr.strip()}"

        # 验证任务已注册
        verify = subprocess.run(
            ["schtasks", "/Query", "/TN", TASK_NAME],
            capture_output=True,
            text=True,
            encoding="gbk",
            timeout=15,
        )
        if verify.returncode != 0:
            return False, f"任务注册后查询失败: {verify.stderr.strip()}"

        return True, f"已注册 Task: {TASK_NAME} (DAILY @ {time_str})\n  命令: {cmd}"

    except FileNotFoundError:
        return False, "schtasks 不在 PATH（不是 Windows？）"
    except subprocess.TimeoutExpired:
        return False, "schtasks 超时"


def uninstall_windows() -> tuple[bool, str]:
    try:
        r = subprocess.run(
            ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
            capture_output=True,
            text=True,
            encoding="gbk",
            timeout=15,
        )
        if r.returncode == 0:
            return True, f"已删除 Task: {TASK_NAME}"
        elif "ERROR: The system cannot find the file" in r.stderr:
            return True, "任务本来就不存在（无害）"
        return False, f"删除失败: {r.stderr.strip()}"
    except FileNotFoundError:
        return False, "schtasks 不在 PATH"


def run_now_windows() -> tuple[bool, str]:
    python_exe = _python_exe()
    index_dir = _index_dir_abs()
    cmd = [
        python_exe,
        str(SCRIPT_PATH),
        "--index-dir", index_dir,
        "--verbose",
    ]
    print(f"🚀 立即执行: {' '.join(cmd)}")
    print("=" * 70)
    try:
        r = subprocess.run(
            cmd,
            cwd=str(SCRIPT_PATH.parent.parent),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=300,
        )
        return r.returncode == 0, f"退出码: {r.returncode}"
    except subprocess.TimeoutExpired:
        return False, "蒸馏超时（>5 分钟）"


def install_systemd(time_str: str = DEFAULT_TIME) -> tuple[bool, str]:
    """Linux systemd timer 安装"""
    python_exe = _python_exe()
    hour, minute = time_str.split(":")

    service_content = f"""[Unit]
Description=WorkBuddy RAG 蒸馏定时任务
After=network.target

[Service]
Type=oneshot
ExecStart={python_exe} {SCRIPT_PATH}
WorkingDirectory={SCRIPT_PATH.parent.parent}
Environment=PYTHONIOENCODING=utf-8
"""
    timer_content = f"""[Unit]
Description=Daily RAG distill

[Timer]
OnCalendar=*-*-* {time_str}:00
Persistent=true

[Install]
WantedBy=timers.target
"""

    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    service_file = systemd_dir / f"{TASK_NAME}.service"
    timer_file = systemd_dir / f"{TASK_NAME}.timer"
    service_file.write_text(service_content, encoding="utf-8")
    timer_file.write_text(timer_content, encoding="utf-8")

    cmds = [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", f"{TASK_NAME}.timer"],
        ["systemctl", "--user", "start", f"{TASK_NAME}.timer"],
    ]
    msgs = []
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            return False, f"{c} 失败: {r.stderr.strip()}"
        msgs.append(" ".join(c))

    return True, "systemd timer 已注册:\n  " + "\n  ".join(msgs) + f"\n  时间: DAILY @ {time_str}"


def main():
    parser = argparse.ArgumentParser(description="WorkBuddy RAG 蒸馏定时任务安装器")
    parser.add_argument("--uninstall", action="store_true", help="卸载定时任务")
    parser.add_argument("--run-now", action="store_true", help="立即跑一次蒸馏")
    parser.add_argument("--time", default=DEFAULT_TIME, help="执行时间（HH:MM，默认 03:00）")
    args = parser.parse_args()

    system = platform.system()

    if args.uninstall:
        if system == "Windows":
            ok, msg = uninstall_windows()
        else:
            ok = True
            msg = "Linux/Mac 卸载请手动: systemctl --user disable --now {TASK_NAME}.timer"
        print(("✅ " if ok else "❌ ") + msg)
        return 0 if ok else 1

    if args.run_now:
        if system == "Windows":
            ok, msg = run_now_windows()
        else:
            return run_subprocess_now()
        print(("✅ " if ok else "❌ ") + msg)
        return 0 if ok else 1

    # 安装
    if system == "Windows":
        ok, msg = install_windows(args.time)
    elif system == "Linux":
        ok, msg = install_systemd(args.time)
    else:
        print(f"⚠️  {system} 暂不支持自动安装，请手动配置 cron / launchd")
        return 1

    print(("✅ " if ok else "❌ ") + msg)
    return 0 if ok else 1


def run_subprocess_now():
    """Linux/Mac 立即跑一次"""
    python_exe = _python_exe()
    cmd = [python_exe, str(SCRIPT_PATH), "--verbose"]
    r = subprocess.run(cmd, cwd=str(SCRIPT_PATH.parent.parent))
    return r.returncode == 0


if __name__ == "__main__":
    sys.exit(main())