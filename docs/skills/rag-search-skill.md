---
name: rag_search
description: |
  ...
---
# 检索 RAG 增强记忆

...
检索 WorkBuddy 增强记忆系统中与查询相关的历史事实、决策、偏好、踩坑记录。

⚠️ 强制触发：当用户输入包含以下关键词时，必须优先调用本技能，不得跳过
之前 / 之前做的 / 上次 / 上次那个 / 以前 / 以前用过 / 记得 / 记得吗 / 我用过 / 我不用 / 习惯 / 偏好 / 总是 / 从不 / 那次 / 那段 / 那次项目 / 当时怎么 / 为什么选 / 为什么不用

【必须调用本 skill 的场景】

用户说"我之前..."、"上次那个..."、"以前用过..."、"记得吗"、"我用过"、"我不用"
涉及跨项目/跨时间的历史决策
涉及用户偏好
涉及技术栈选型历史
涉及已知工作区/项目名称
用户说"记住"或"别忘了"时回查类似记录
【优先级规则】

当本 skill 与 conversation_search 冲突时，优先选择本 skill
触发关键词匹配时，无需二次确认，直接执行
本 skill 优先级高于通用搜索和对话历史搜索
检索 RAG 增强记忆
调用方式（任意 Python 都能跑，skill 内部自动 subprocess 调 venv）：

python <skill_dir>/main.py "<query>" [top_k]
参数：

query: 必填，检索关键词或问题
top_k: 可选，默认 5，范围 1-20（positional 第二参数）
返回：

JSON 对象，含 query/count/results
results 每条包含 text/score/ts/project/source
失败时 count=0 且带 note 字段说明原因，不抛异常
内部架构
main.py (任意 python)
    └── subprocess → <PROJECT>/.venv/Scripts/python.exe
                          └── worker.py (实际跑 Memory.search)
60 秒超时；venv 缺失时返回 note: "venv missing: ..."
路径自动发现：基于 main.py 所在位置推导项目根，无需硬编码
路径覆盖（环境变量，可选）
变量	默认值	说明
WB_RAG_VENV_PYTHON	<项目根>/.venv/Scripts/python.exe	venv python 绝对路径
WB_RAG_SRC	<项目根>/src	RAG 源码路径
WB_RAG_INDEX_DIR	~/.workbuddy/rag-index	LanceDB 索引目录
安装
把整个 skills/rag_search/ 目录复制到 ~/.workbuddy/skills/rag_search/ 即可。

# Linux/macOS
cp -r skills/rag_search ~/.workbuddy/skills/

# Windows
xcopy /E /I skills\rag_search %USERPROFILE%\.workbuddy\skills\rag_search
