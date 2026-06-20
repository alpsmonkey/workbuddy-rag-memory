---
name: rag_search
description: |
  检索 WorkBuddy 增强记忆系统中与查询相关的历史事实、决策、偏好、踩坑记录。

  【必须调用本 skill 的场景】
  - 用户说"我之前..."、"上次那个..."、"以前用过..."、"记得吗"、"我用过"、"我不用"
  - 涉及跨项目/跨时间的历史决策（"为什么当时选 X 不选 Y"、"我们之前怎么解决的"）
  - 涉及用户偏好（"我习惯..."、"我偏好..."、"我不喜欢..."、"我喜欢..."）
  - 涉及技术栈选型历史（"我们用什么..."、"上次用什么技术"）
  - 涉及已知工作区/项目名称（"SkillFather"、"成都3日游"、"WorkBuddy"、"Vibe Coding"等）
  - 用户说"记住"或"别忘了"时回查类似记录
  - 用户说"我之前在...遇到过..."、"上次那种情况怎么处理的"

  【不要调用的场景】
  - 当前会话内已讨论的问题（避免重复查询）
  - 通用知识问答（"Python 是什么"）
  - 实时性要求高的查询（"现在几点"、"今天日期"）
  - 一次性技术问题（"怎么写 SQL"）

  【触发关键词速查】
  之前/上次/以前/记得/我用过/我不用/习惯/偏好/总是/从不
  为什么选/为什么不用/当时怎么/那次/那段/那次项目
---

# 检索 RAG 增强记忆

调用方式（**任意 Python 都能跑**，skill 内部自动 subprocess 调 venv）：
```
python <skill_dir>/main.py "<query>" [top_k]
```

参数：
- query: 必填，检索关键词或问题
- top_k: 可选，默认 5，范围 1-20（positional 第二参数）

返回：
- JSON 对象，含 `query`/`count`/`results`
- `results` 每条包含 `text`/`score`/`ts`/`project`/`source`
- 失败时 `count=0` 且带 `note` 字段说明原因，不抛异常

## 内部架构

```
main.py (任意 python)
    └── subprocess → <PROJECT>/.venv/Scripts/python.exe
                          └── worker.py (实际跑 Memory.search)
```

- **60 秒超时**；venv 缺失时返回 `note: "venv missing: ..."`
- **路径自动发现**：基于 main.py 所在位置推导项目根，无需硬编码

## 路径覆盖（环境变量，可选）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WB_RAG_VENV_PYTHON` | `<项目根>/.venv/Scripts/python.exe` | venv python 绝对路径 |
| `WB_RAG_SRC` | `<项目根>/src` | RAG 源码路径 |
| `WB_RAG_INDEX_DIR` | `~/.workbuddy/rag-index` | LanceDB 索引目录 |

## 安装

把整个 `skills/rag_search/` 目录复制到 `~/.workbuddy/skills/rag_search/` 即可。

```bash
# Linux/macOS
cp -r skills/rag_search ~/.workbuddy/skills/

# Windows
xcopy /E /I skills\rag_search %USERPROFILE%\.workbuddy\skills\rag_search
```