"""
install_skills.py — 自动同步项目 skill 到 user-level ~/.workbuddy/skills/

动机：
  - 项目内的 skill 是"源码"（路径自动发现 + 随 Git 同步）
  - user-level 的 skill 是"部署目标"（WorkBuddy 实际加载的位置）
  - 修了代码后需要手动 cp -r，容易忘 → 此脚本自动化

同步策略：
  - 完全覆盖（源码优先）：项目版本 = 权威
  - 不删 user-level 已有的其他 skill（只覆盖同名项）
  - 支持 --dry-run 预览、--verbose 详细输出

当前同步的 skill：
  - rag_search/（RAG 增强记忆检索）

用法:
  python scripts/install_skills.py              # 安装到 ~/.workbuddy/skills/
  python scripts/install_skills.py --dry-run    # 预览不写
  python scripts/install_skills.py --verbose    # 显示详细
  python scripts/install_skills.py --target C:/other/skills/  # 自定义目标
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path


# 从脚本位置推导项目根（与 skills/rag_search/main.py 同模式）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_SRC = PROJECT_ROOT / "skills"

# 默认目标：WorkBuddy user-level skills 目录
DEFAULT_TARGET = Path.home() / ".workbuddy" / "skills"

# 要同步的 skill 子目录（相对于 SKILLS_SRC）
# 目前只有 rag_search，后续可加 rag_bootstrap 等项目镜像
SKILLS_TO_INSTALL = [
    "rag_search",
]


def _log(msg: str, verbose: bool = False, level: str = "info") -> None:
    """统一日志输出"""
    prefix = {"info": "  ", "ok": "✅", "skip": "⏭️", "warn": "⚠️", "error": "❌"}.get(level, "  ")
    if verbose or level in ("ok", "error", "warn"):
        print(f"{prefix} {msg}")


def install_one(src_dir: Path, target_dir: Path, dry_run: bool = False, verbose: bool = False) -> dict:
    """安装单个 skill：清空目标 → 复制全量源文件

    Returns:
        {"name": str, "files_copied": int, "skipped": bool, "error": str|None}
    """
    if not src_dir.exists():
        return {"name": src_dir.name, "files_copied": 0, "skipped": False, "error": "源目录不存在"}

    # 收集源文件列表
    src_files = list(src_dir.rglob("*"))
    py_files = [f for f in src_files if f.is_file()]

    if dry_run:
        _log(f"{src_dir.name}/ → {target_dir}/ ({len(py_files)} files)", verbose, "info")
        return {"name": src_dir.name, "files_copied": len(py_files), "skipped": False, "error": None}

    # 清空目标（如果有旧版本）
    if target_dir.exists():
        _log(f"覆盖: {target_dir}", verbose)
        shutil.rmtree(str(target_dir))

    target_dir.mkdir(parents=True, exist_ok=True)

    # 复制所有文件（保持目录结构）
    copied = 0
    for src_file in src_files:
        if src_file.is_dir():
            (target_dir / src_file.relative_to(src_dir)).mkdir(parents=True, exist_ok=True)
            continue
        dst = target_dir / src_file.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src_file), str(dst))
        copied += 1
        if verbose:
            _log(f"  {src_file.relative_to(src_dir)}", verbose, "info")

    return {"name": src_dir.name, "files_copied": copied, "skipped": False, "error": None}


def install_all(target: Path, dry_run: bool = False, verbose: bool = False, skills: list = None) -> int:
    """批量安装

    Returns:
        0 = 成功, 1 = 部分失败
    """
    if skills is None:
        skills = SKILLS_TO_INSTALL

    if not SKILLS_SRC.exists():
        _log(f"项目 skills 目录不存在: {SKILLS_SRC}", verbose, "error")
        return 1

    results = []
    for skill_name in skills:
        src_dir = SKILLS_SRC / skill_name
        target_dir = target / skill_name
        r = install_one(src_dir, target_dir, dry_run=dry_run, verbose=verbose)
        results.append(r)

    # 汇总报告
    total_files = sum(r["files_copied"] for r in results)
    errors = [r for r in results if r.get("error")]

    if dry_run:
        print(f"\n[DRY-RUN] 将安装 {len(results)} 个 skill（{total_files} 文件）到 {target}")
        for r in results:
            if r.get("error"):
                _log(f"{r['name']}: {r['error']}", True, "error")
    else:
        ok_count = len([r for r in results if not r.get("error")])
        print(f"\n完成: {ok_count}/{len(results)} skill 已安装, {total_files} 文件")
        for r in results:
            if r.get("error"):
                _log(f"{r['name']}: {r['error']}", True, "error")
            elif r["files_copied"]:
                _log(f"{r['name']}: {r['files_copied']} files", verbose, "ok")

    return 0 if not errors else 1


def main():
    parser = argparse.ArgumentParser(
        description="安装项目 skill 到 WorkBuddy user-level skills 目录"
    )
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET),
                        help=f"目标 skills 目录（默认 {DEFAULT_TARGET}）")
    parser.add_argument("--dry-run", action="store_true", help="预览不写")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出每个文件")
    parser.add_argument("--skill", type=str, nargs="*",
                        help=f"只安装指定 skill（默认全部: {', '.join(SKILLS_TO_INSTALL)}）")
    args = parser.parse_args()

    target = Path(args.target)
    skills_to_install = args.skill if args.skill else SKILLS_TO_INSTALL

    if args.verbose:
        print(f"项目根: {PROJECT_ROOT}")
        print(f"源: {SKILLS_SRC}")
        print(f"目标: {target}")
        print(f"模式: {'DRY-RUN' if args.dry_run else '实际写入'}")
        print(f"Skill: {', '.join(skills_to_install)}")
        print()

    sys.exit(install_all(target, dry_run=args.dry_run, verbose=args.verbose, skills=skills_to_install))


if __name__ == "__main__":
    main()
