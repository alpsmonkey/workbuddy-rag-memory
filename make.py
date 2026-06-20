"""
make 命令在 Windows 上不通用（默认没装），提供 Python 替代入口

用法:
  python make.py health
  python make.py ingest
  python make.py query "SkillFather 是什么"
  python make.py distill
  python make.py bootstrap
  python make.py test
  python make.py help

等价于 `make <target>`（Linux/Mac 上推荐用 make）
"""
from __future__ import annotations
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def venv_python() -> str:
    """优先用 venv python，否则用当前 python"""
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows venv
        PROJECT_ROOT / ".venv" / "bin" / "python",          # Linux/Mac venv
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


def run(cmd: list, **kwargs) -> int:
    """跑命令并打印"""
    print(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd, **kwargs)


# ============================================================================
# 命令实现
# ============================================================================

def cmd_help():
    print(__doc__)
    return 0


def cmd_install():
    return run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def cmd_install_dev():
    rc = cmd_install()
    if rc != 0:
        return rc
    return run([sys.executable, "-m", "pip", "install", "-e", ".[test,watchdog]"])


def cmd_download_models():
    if not Path.home().joinpath(".cache", "huggingface", "models--BAAI--bge-m3").exists():
        print("Downloading bge-m3...")
        run([venv_python(), "download_bge_m3.py"])
    else:
        print("✓ bge-m3 already cached")

    if not Path.home().joinpath(".cache", "huggingface", "models--BAAI--bge-reranker-v2-m3").exists():
        print("Downloading bge-reranker-v2-m3...")
        run([venv_python(), "download_bge_reranker.py"])
    else:
        print("✓ bge-reranker-v2-m3 already cached")
    return 0


def cmd_health():
    return run([venv_python(), "-m", "scripts.health"], cwd=str(PROJECT_ROOT))


def cmd_ingest():
    return run([venv_python(), "-m", "scripts.ingest_wb_memory", "--verbose"],
               cwd=str(PROJECT_ROOT))


def cmd_ingest_dry():
    return run([venv_python(), "-m", "scripts.ingest_wb_memory", "--dry-run", "--verbose"],
               cwd=str(PROJECT_ROOT))


def cmd_query():
    """make.py query '你的问题'"""
    if len(sys.argv) < 3:
        print("用法: python make.py query '你的问题'")
        return 1
    query = sys.argv[2]
    return run([venv_python(), "-m", "scripts.query", query, "--top-k", "5"],
               cwd=str(PROJECT_ROOT))


def cmd_distill():
    index_dir = str(Path.home() / ".workbuddy" / "rag-index")
    return run([venv_python(), "-m", "scripts.distill", "--index-dir", index_dir, "--verbose"],
               cwd=str(PROJECT_ROOT))


def cmd_distill_cron():
    return run([venv_python(), "-m", "scripts.install_distill_cron"],
               cwd=str(PROJECT_ROOT))


def cmd_distill_cron_uninstall():
    return run([venv_python(), "-m", "scripts.install_distill_cron", "--uninstall"],
               cwd=str(PROJECT_ROOT))


def cmd_bootstrap():
    return run([venv_python(), "-m", "scripts.install_bootstrap"],
               cwd=str(PROJECT_ROOT))


def cmd_bootstrap_uninstall():
    return run([venv_python(), "-m", "scripts.install_bootstrap", "--uninstall"],
               cwd=str(PROJECT_ROOT))


def cmd_test():
    return run([venv_python(), "-m", "pytest", "tests/", "-v"], cwd=str(PROJECT_ROOT))


def cmd_clean():
    import shutil
    for pattern in [".index", ".pytest_cache", "**/__pycache__"]:
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)
    for p in PROJECT_ROOT.rglob("*.pyc"):
        p.unlink(missing_ok=True)
    print("✓ cleaned")
    return 0


# ============================================================================
# 入口
# ============================================================================

COMMANDS = {
    "help": cmd_help,
    "install": cmd_install,
    "install-dev": cmd_install_dev,
    "download-models": cmd_download_models,
    "health": cmd_health,
    "ingest": cmd_ingest,
    "ingest-dry": cmd_ingest_dry,
    "query": cmd_query,
    "distill": cmd_distill,
    "distill-cron": cmd_distill_cron,
    "distill-cron-uninstall": cmd_distill_cron_uninstall,
    "bootstrap": cmd_bootstrap,
    "bootstrap-uninstall": cmd_bootstrap_uninstall,
    "test": cmd_test,
    "clean": cmd_clean,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        cmd_help()
        return 0

    target = sys.argv[1]
    if target not in COMMANDS:
        print(f"❌ 未知命令: {target}")
        print(f"可用: {', '.join(COMMANDS.keys())}")
        return 1

    return COMMANDS[target]()


if __name__ == "__main__":
    sys.exit(main())