# WorkBuddy Skills

把 `~/.workbuddy/skills/` 下的 RAG 相关 skill **镜像**到本目录，方便：
- 版本跟随项目走（`git pull` 就拿到最新修复）
- 跨机器同步（clone 项目即可获得 skill 源码）
- 不依赖个人绝对路径（路径通过 `__file__` 自动发现）

## 当前包含的 skill

| Skill | 用途 | 文档 |
|---|---|---|
| `rag_search/` | 检索 RAG 增强记忆（按 query 找历史 chunk）| [SKILL.md](rag_search/SKILL.md) |

## 安装到 WorkBuddy

把整个 skill 目录复制到 `~/.workbuddy/skills/` 即可（覆盖即可生效）：

```bash
# Windows (Git Bash)
cp -r skills/rag_search ~/.workbuddy/skills/

# Windows (PowerShell)
Copy-Item -Recurse -Force skills\rag_search $env:USERPROFILE\.workbuddy\skills\

# Linux / macOS
cp -r skills/rag_search ~/.workbuddy/skills/
```

或者用 `scripts/install_bootstrap.py` 风格写个 `install_skills.py`（TODO）。

## 路径自动发现

所有 skill 都通过 `Path(__file__).resolve().parent.parent` 推导**项目根**，从而避免硬编码 `E:\workspace\...` 这种个人路径。

```
<PROJECT>/                     ← 项目根
├── .venv/Scripts/python.exe   ← venv python
├── src/                       ← RAG 源码
└── skills/rag_search/         ← 本目录
    ├── SKILL.md
    ├── main.py                ← 任意 python 都能调（subprocess 隔离）
    └── worker.py              ← 实际跑 Memory.search（venv python 调）
```

可覆盖的环境变量：
- `WB_RAG_VENV_PYTHON` 指定 venv python 绝对路径
- `WB_RAG_SRC` 指定 src 绝对路径
- `WB_RAG_INDEX_DIR` 指定索引目录（默认 `~/.workbuddy/rag-index`）