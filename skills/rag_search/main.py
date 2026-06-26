"""
rag_search skill - subprocess 模式（自动发现路径版）

WorkBuddy 调 skill 时用什么 python 都不影响——skill 自己 subprocess 调 venv python。

调用方式（WorkBuddy 任意 python 都能调）：
  python main.py "<query>" [top_k]
  python main.py --stdin   # 从 stdin 读 JSON {"query": "...", "top_k": 5}

路径自动发现：
  本文件所在位置: <PROJECT>/skills/rag_search/main.py
  → PROJECT = 父目录的父目录
  → VENV = PROJECT/.venv/Scripts/python.exe
  → WORKER = 本文件同目录/worker.py
  → SRC = PROJECT/src（注入 sys.path）

可覆盖：
  WB_RAG_VENV_PYTHON   指定 venv python 绝对路径
  WB_RAG_INDEX_DIR     指定索引目录（默认 ~/.workbuddy/rag-index）
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# 路径自动发现（多策略：避免硬编码个人绝对路径，便于开源分发）
# ============================================================
# 优先级（从高到低）：
#   1. config.json（与 main.py 同目录，由 install_skills.py 自动生成）
#   2. 自动发现：PROJECT_ROOT/.venv 存在 → 当作项目根
#   3. 回退：env var（WB_RAG_VENV_PYTHON / WB_RAG_INDEX_DIR / WB_RAG_SRC）
# 用途：skill 安装到 ~/.workbuddy/skills/ 后，自动发现会错；
#       config.json 含项目根绝对路径，规避 SKILL_DIR.parent.parent 推导错误。
# ============================================================
_THIS_FILE = Path(__file__).resolve()
SKILL_DIR = _THIS_FILE.parent                  # skills/rag_search/
SKILLS_ROOT = SKILL_DIR.parent                 # skills/
PROJECT_ROOT = SKILLS_ROOT.parent              # 项目根（含 src/ 和 .venv/），仅在源位置时正确

# 默认 venv python（项目根/.venv/Scripts/python.exe）
_DEFAULT_VENV = PROJECT_ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

# worker.py 跟 main.py 同目录
WORKER_PATH = SKILL_DIR / "worker.py"

# 超时（秒）— 首次冷启动要加载 bge-m3 到内存，60-70s 实测不够，给 180s
TIMEOUT_SEC = 180


def _discover_paths():
    """多策略路径发现。Returns: (project_root: Path|None, index_dir: Path)

    - 找到 config.json：用其声明的绝对路径（无 tilde 隐患）
    - 自动发现成立（PROJECT_ROOT/.venv 存在）：用推导路径
    - 都没：project_root=None，让 run_search 走 env var 回退
    """
    # 1. config.json（最高优先级，含绝对路径）
    config_path = SKILL_DIR / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            return Path(cfg["project_root"]), Path(cfg["index_dir"])
        except Exception as e:
            print(f"[main] config.json 解析失败，回退: {e}", file=sys.stderr)

    # 2. 自动发现：项目根必须有 .venv 子目录（作为锚点）
    if (PROJECT_ROOT / ".venv").exists():
        return PROJECT_ROOT, Path.home() / ".workbuddy" / "rag-index"

    # 3. 回退：依赖 env var
    return None, Path(os.environ.get("WB_RAG_INDEX_DIR", str(Path.home() / ".workbuddy" / "rag-index")))


def run_search(query: str, top_k: int = 5, use_hyde: bool = True) -> dict:
    """subprocess 调 venv python 跑检索

    Args:
        query: 检索 query
        top_k: 返回条数
        use_hyde: 是否启用 HyDE Query 改写（默认 True，Mock 模式零额外资源）
    """
    if not query:
        return {"query": "", "count": 0, "results": [], "note": "empty query"}

    # 多策略路径发现
    project_root, index_dir = _discover_paths()

    # venv python 决策
    if project_root is not None:
        venv = project_root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    else:
        venv = Path(os.environ.get("WB_RAG_VENV_PYTHON", str(_DEFAULT_VENV)))

    if not venv.exists():
        return {"query": query, "count": 0, "results": [], "note": f"venv missing: {venv}"}
    if not WORKER_PATH.exists():
        return {"query": query, "count": 0, "results": [], "note": f"worker missing: {WORKER_PATH}"}

    cmd = [str(venv), str(WORKER_PATH), query, str(top_k)]
    if not use_hyde:
        cmd.append("--no-hyde")

    # 关键：把项目根 src 和 index_dir 通过 env 传给 worker（worker 也读这俩 env var）
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",
    }
    if project_root is not None:
        env["WB_RAG_SRC"] = str(project_root / "src")
    env["WB_RAG_INDEX_DIR"] = str(index_dir)

    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=TIMEOUT_SEC,
        )
        if r.returncode != 0:
            return {
                "query": query, "count": 0, "results": [],
                "note": f"worker exit={r.returncode}: {(r.stderr or '')[-200:]}",
            }
        # worker 输出就是 JSON
        return json.loads(r.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"query": query, "count": 0, "results": [], "note": f"timeout (>{TIMEOUT_SEC}s)"}
    except Exception as e:
        return {"query": query, "count": 0, "results": [], "note": f"error: {e}"}


def format_output(result: dict) -> str:
    """格式化输出：JSON"""
    if not result.get("results"):
        return json.dumps(
            {**result, "note": result.get("note", "未找到相关记忆或检索失败")},
            ensure_ascii=False,
        )
    return json.dumps(result, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="RAG 增强记忆检索")
    parser.add_argument("query", nargs="?", help="检索 query")
    parser.add_argument("top_k_pos", nargs="?", type=int, default=None, help="top_k (positional)")
    parser.add_argument("--top-k", type=int, default=5, help="返回条数（1-20）")
    parser.add_argument("--no-hyde", dest="hyde", action="store_false", default=True,
                        help="禁用 HyDE Query 改写（默认启用 Mock HyDE，零额外资源）")
    parser.add_argument("--stdin", action="store_true", help="从 stdin 读 JSON")
    args = parser.parse_args()

    if args.stdin:
        try:
            data = json.loads(sys.stdin.read())
            query = data.get("query", "").strip()
            top_k = data.get("top_k", 5)
        except Exception:
            print(format_output({"query": "", "count": 0, "results": []}))
            return
    elif args.query:
        query = args.query.strip()
        top_k = args.top_k_pos if args.top_k_pos is not None else args.top_k
    else:
        print(format_output({"query": "", "count": 0, "results": []}), file=sys.stderr)
        sys.exit(1)

    top_k = max(1, min(20, top_k))
    result = run_search(query, top_k, use_hyde=args.hyde)
    print(format_output(result))


if __name__ == "__main__":
    main()