#!/bin/bash
# WorkBuddy RAG Memory - 推送到 GitHub 脚本
# 用法：
#   GITHUB_TOKEN=ghp_xxx ./push_to_github.sh <github_user> [<repo_name>]
#
# 示例：
#   GITHUB_TOKEN=ghp_xxx ./push_to_github.sh peng42
#   GITHUB_TOKEN=ghp_xxx ./push_to_github.sh your-org workbuddy-rag-memory

set -e

GITHUB_USER="${1:-your-username}"
REPO_NAME="${2:-workbuddy-rag-memory}"
GITHUB_TOKEN="${GITHUB_TOKEN:?必须设置 GITHUB_TOKEN 环境变量}"

echo "🚀 推送 workbuddy-rag-memory → https://github.com/${GITHUB_USER}/${REPO_NAME}"

# 1. 创建远程 repo（GitHub API）
echo "📡 检查/创建 GitHub repo..."
HTTP_CODE=$(curl -s -o /tmp/gh_create.json -w "%{http_code}" \
    -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/user/repos" \
    -d "{\"name\":\"${REPO_NAME}\",\"description\":\"WorkBuddy RAG 增强记忆系统：三索引 + 去重 + 时间衰减 + Cross-Encoder 重排 + 自动蒸馏\",\"private\":false,\"has_issues\":true,\"has_wiki\":false}")

if [ "$HTTP_CODE" = "201" ]; then
    echo "✅ repo 创建成功"
elif [ "$HTTP_CODE" = "422" ]; then
    echo "⚠️  repo 已存在（422），继续推送"
else
    echo "❌ repo 创建失败 HTTP $HTTP_CODE"
    cat /tmp/gh_create.json
    exit 1
fi

# 2. 添加 remote + 推送
git remote remove origin 2>/dev/null || true
git remote add origin "https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "📤 git push -u origin main..."
git push -u origin main --force 2>&1 | tail -10

echo ""
echo "✅ 推送完成！"
echo "   Repo URL: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "   Clone:    git clone https://github.com/${GITHUB_USER}/${REPO_NAME}.git"