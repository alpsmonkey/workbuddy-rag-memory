"""
WorkBuddy RAG Memory - 推送到 GitHub 脚本（Windows 版）
用法：
  set GITHUB_TOKEN=ghp_xxx
  python push_to_github.py <github_user> [<repo_name>]

示例：
  set GITHUB_TOKEN=ghp_xxx
  python push_to_github.py peng42
  python push_to_github.py your-org workbuddy-rag-memory
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from urllib import request
from urllib.error import HTTPError
import json


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    github_user = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else "workbuddy-rag-memory"
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        print("❌ 必须设置环境变量 GITHUB_TOKEN")
        print("   set GITHUB_TOKEN=ghp_xxxxxxxxxxxx")
        sys.exit(1)

    print(f"🚀 推送 workbuddy-rag-memory → https://github.com/{github_user}/{repo_name}")

    # 1. 创建/确认远程 repo
    print("📡 检查/创建 GitHub repo...")
    url = "https://api.github.com/user/repos"
    payload = json.dumps({
        "name": repo_name,
        "description": "WorkBuddy RAG 增强记忆系统：三索引 + 去重 + 时间衰减 + Cross-Encoder 重排 + 自动蒸馏",
        "private": False,
        "has_issues": True,
        "has_wiki": False,
        "auto_init": False,
    }).encode("utf-8")

    req = request.Request(
        url, data=payload, method="POST",
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        }
    )

    try:
        with request.urlopen(req) as resp:
            print(f"✅ repo 创建成功 (HTTP {resp.status})")
            print(f"   {resp.read().decode()[:200]}")
    except HTTPError as e:
        if e.code == 422:
            print("⚠️  repo 已存在（422），继续推送")
        else:
            print(f"❌ repo 创建失败 HTTP {e.code}")
            print(e.read().decode())
            sys.exit(1)

    # 2. 设置 remote + push
    remote_url = f"https://{github_token}@github.com/{github_user}/{repo_name}.git"

    # 删除已存在的 origin（幂等）
    subprocess.run(["git", "remote", "remove", "origin"], capture_output=True)

    subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
    print(f"📤 git push -u origin main...")
    r = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], capture_output=True, text=True)

    print(r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        sys.exit(1)

    print()
    print("✅ 推送完成！")
    print(f"   Repo URL: https://github.com/{github_user}/{repo_name}")
    print(f"   Clone:    git clone https://github.com/{github_user}/{repo_name}.git")


if __name__ == "__main__":
    main()