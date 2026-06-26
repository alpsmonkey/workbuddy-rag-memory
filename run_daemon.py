"""
RAG 守护进程启动器（薄包装）

解决问题:
  - 任务调度器 / 双击 run.bat 时, cwd 不可控 → sys.path 找不到 src/
  - 旧 daemon.py 硬编码 3388 路径, 换了 workspace 就挂
  - 用本文件做唯一入口, 路径全部按 __file__ 动态推导

用法:
  python run_daemon.py            # 控制台模式
  pythonw.exe run_daemon.py       # 无窗口模式 (run.bat 用这个)
"""
from __future__ import annotations
import sys
from pathlib import Path

# === 路径全部按 __file__ 推导, 不再硬编码 ===
HERE = Path(__file__).resolve().parent          # .../workbuddy-rag-memory/
SRC_DIR = HERE / "src"                           # .../workbuddy-rag-memory/src
DAEMON_DIR = Path.home() / ".workbuddy" / "rag-daemon"   # C:\Users\JJ\.workbuddy\rag-daemon
DAEMON_PY = DAEMON_DIR / "daemon.py"

# 1) 把 src 加进 sys.path, 让 memory.py 能 import
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 2) 切到项目根, 相对路径都按这里算
import os
os.chdir(str(HERE))

# 3) 直接 exec daemon.py（不依赖 import 路径, 避免重复 sys.path 注入)
if __name__ == "__main__":
    if not DAEMON_PY.exists():
        print(f"[FATAL] daemon.py 不存在: {DAEMON_PY}")
        sys.exit(2)
    # 用 runpy 跑, 让 daemon 里的 if __name__ == "__main__" 走 main()
    import runpy
    runpy.run_path(str(DAEMON_PY), run_name="__main__")
