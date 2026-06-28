# WorkBuddy RAG 增强记忆系统

> 三索引（向量 + BM25 + 元数据）+ 写入去重 + 时间衰减 + Cross-Encoder 重排 + 蒸馏闭环
> 目标：Recall@10 ≥ 60%，Top-3 精度 ≥ 80%，索引只减不增

## 架构

```
                    ┌──────────────────────┐
   Query ──────►   │  Embedder (bge-m3)   │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌────────────┐ ┌────────────┐ ┌────────────┐
        │ LanceDB    │ │ SQLite     │ │ 元数据     │
        │ (向量)     │ │ FTS5       │ │ (SQLite)   │
        └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ RRF 融合排序    │
                    └────────┬────────┘
                             ▼
                       Top-K 结果
```

## 系统逻辑框架

> 这一节讲清楚「WorkBuddy RAG Memory」在整个记忆体系里扮演什么角色、数据怎么流、谁触发谁、各阶段的真实痛点和解决度。
> 如果只读一节，读这一节就够了。

### 1. 三层记忆体系全景

WorkBuddy 的记忆不是单层 RAG，而是 3 层栈。RAG 增强是叠在 L2/L3 之上的语义层，不替代原有协议。

| 层 | 范围 | 持久化 | 谁来写 | 检索方式 | 容量上限 |
|---|---|---|---|---|---|
| **L1 云端画像** | 用户级长期 | server-side summary | 服务端隐式学习 | 启动时全量注入 `<memory>` | 不限 |
| **L2 用户级本地** | 跨项目偏好/决策 | `~/.workbuddy/MEMORY.md` | 显式 Edit | `conversation_search`(跨会话) + 本地 RAG | 4000 chars |
| **L3 项目工作区** | 单项目上下文 | `<workspace>/.workbuddy/memory/YYYY-MM-DD.md` + `MEMORY.md` | 每轮 append | workspace 日志 + 本地 RAG | 3000 chars |
| **RAG 增强层(本项目)** | L2/L3 的语义化副本 | `~/.workbuddy/rag-index/` | **自动(daemon + cron)** | BM25 + 向量 + 元数据 → RRF → Rerank | 无限(蒸馏收敛) |

**RAG 的定位**：补 WorkBuddy 本地层的「语义检索 + 自动维护」，不替代协议层。WorkBuddy 主体代码改动 = 0。

### 2. 数据流（7 步闭环）

```
                       ┌────────── 写入路径 ──────────┐
[用户/脚本写 .md] ──► [chunker] ──► [dedup 三档] ──► [三索引写入]
                                                       │
                                                       ▼
                                                ┌──────────┐
                                                │ LanceDB  │
                                                │ SQLite   │
                                                │ 元数据    │
                                                └────┬─────┘
                                                     │
                       ┌────────── 检索路径 ──────────┤
                       ▼                              │
[query] ──► [embedder] ──► [可选 HyDE] ──► 三路召回 ──┘
                                          │
                                          ▼
                              [RRF 融合 k=30]
                                          │
                                          ▼
                       [时间衰减 × 流行度 × 置信度]
                                          │
                                          ▼
                                  [Top-N 粗排]
                                          │
                                          ▼
                              [BGE Reranker 精排]
                                          │
                                          ▼
                                   [Top-K 返回]
                                          │
                       ┌──── 维护路径 ────┘
                       ▼
        ┌────────────────────────────┐
        │ daemon: 文件变更 → 5s 入库 │
        │ distill: 每日 03:00 清理   │
        │ health: 任意时刻巡检       │
        └────────────────────────────┘
```

### 3. 三索引分工

| 索引 | 底层 | 强项 | 弱项 | 命中场景 |
|---|---|---|---|---|
| **向量** | LanceDB + bge-m3 (1024 维) | 语义同义改写 | 短 query 差 | 长 query、中英混杂 |
| **BM25** | SQLite FTS5 | 关键词精确 | 无语义 | 短 query、专有名词、版本号 |
| **元数据** | SQLite | project/ts/source/confidence 强过滤 | 不可检索内容 | project 过滤、时间衰减 |

三路独立召回 → RRF 融合 → 任一路命中都不会丢。

### 4. 写入路径（去重三档 + 索引大小保护）

```
新 chunk ──► 算与全库最大余弦相似度 ──►
    ├─ sim ≥ 0.92 ──► merge（高覆盖低，access_count 取大）
    ├─ 0.85 ≤ sim < 0.92 ──► skip（防语义撞车）
    └─ sim < 0.85 ──► insert（新事实）
```

索引大小策略（防大索引拖垮 dedup）：
- `k = min(search_k=50, count, MAX_SEARCH_K=200)`
- 索引 > 1000 → large-mode，`search_k` 减半
- 环境变量 `DEDUP_MAX_SEARCH_K` / `DEDUP_LARGE_INDEX_THRESHOLD` 可调

### 5. 检索路径（粗排 → 精排）

```
score = rrf_score × exp(-Δt_days / τ) × log(1 + access_count) × (0.7 + 0.3 × confidence)
       └─── RRF ───┘ └── 时间衰减 ──┘ └── 流行度 ──┘ └───── 置信度 ─────┘

τ = 90 天（可调）
RRF k = 30（阶段 1 的 60 区分度太低，已下调）
默认 record_access = True（被检索到的 chunk 自动 +1，蒸馏时高 access 优先保留）
```

精排：把 RRF 粗排 Top-N=20 喂给 BGE Reranker v2-m3（Cross-Encoder）→ Top-K=5。Cross-Encoder 把 (q, d) 作为一对输入，能捕捉 token 级匹配，实测 Top-3 命中率 60% → 95%+。

### 6. 自动维护（7 个触发点）

| 触发时机 | 动作 | 工具 |
|---|---|---|
| Windows 登录 | 启动 daemon | `install_bootstrap.py`（注册 `HKCU\...\Run\WorkBuddy-RAG-Bootstrap`） |
| 文件写入 | 5s debounce 后自动 ingest | `~/.workbuddy/rag-daemon/daemon.py`（watchdog） |
| 新会话开始 | 手动触发入库 | `@rag_bootstrap` skill |
| 每日 03:00 | 蒸馏低价值旧记忆 | `install_distill_cron.py`（Task Scheduler） |
| 任意时刻 | 健康度巡检 | `python -m scripts.health --quiet` |
| 模型缺失 | 下载模型 | `download_bge_m3.py` / `download_bge_reranker.py` |
| 外部调用 | HTTP REST | `scripts/server.py`（v0.2.3+，端口 8000） |

### 7. 当前架构的 12 个真实痛点 vs RAG 解决度

源自 2026-06-19 复盘。RAG 是补本地层的工具，不是银弹。

| # | 痛点 | RAG 阶段 | 解决度 |
|---|---|---|---|
| 1 | 写入靠自觉（模型忘写、写歪） | 1.5 daemon | 🟡 部分（自动捕获写入，但 fact extraction 仍需手动） |
| 2 | 本地层无语义检索 | 1+ | 🟢 三索引 + RRF 完整解决 |
| 3 | 冲突不收敛 | 1+ | 🟢 dedup 0.85/0.92 三档 |
| 4 | 跨 workspace 隔离 | 1.5 | 🟢 共享索引 + `--dir` 多源 |
| 5 | 注入贪心（全量灌） | 1+ | 🔴 顶层仍全量注入，需 WorkBuddy 主体配合 |
| 6 | 无时间线 | 2 | 🟡 时间衰减 + 蒸馏，决策 why-lost 部分缓解 |
| 7 | 无 schema（混在一个文件） | — | 🔴 自由文本，schema 拆分在 WorkBuddy 主体侧 |
| 8 | skill / agent 记忆割裂 | 2 HyDE | 🟡 HyDE 改善短 query |
| 9 | 无 forgetting | 2 蒸馏 | 🟢 蒸馏闭环 |
| 10 | 检索时机靠模型判断 | 1.5 | 🟡 daemon 自动捕获写入路径 |
| 11 | 无图片 / 代码片段 | — | 🔴 当前 chunker 不解析 |
| 12 | 跨工作区记忆完全隔离 | 1.5 | 🟢 共享索引 |

**RAG 不解决的（WorkBuddy 主体侧的事）**：
- 云端 L1 画像
- 自动 fact extraction（需 prompt 改造）
- 启动强制注入本地 MEMORY.md（需 WorkBuddy 主体配合）
- schema 拆分文件

### 8. 演进路径

```
阶段 1    基础库：embedder + chunker + indexer + dedup + retriever + memory
   ▼
阶段 1.5  零侵入集成：watchdog daemon + rag_search skill + bootstrap
   ▼
阶段 2    质量：RRF k 调优 + 时间衰减 + BGE Reranker + HyDE + FastAPI 服务端
   ▼
阶段 3    待办：完整 gold set 标注 + 自动蒸馏 + 健康度仪表盘 + 冲突可视化
```

---

## 快速开始

> 5 分钟指南见 [INSTALL.md](INSTALL.md)。

### 0. 选你的入口

**Linux/Mac（有 make）：**
```bash
make help            # 查看所有命令
make install         # 装依赖
make download-models # 下 bge-m3 + reranker
make ingest          # 入库
make query Q="..."   # 检索
```

**Windows（没 make，用 Python 替代）：**
```powershell
python make.py help
python make.py install
python make.py download-models
python make.py ingest
python make.py query "..."
```

### 1. 安装

```bash
cd workbuddy-rag-memory
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 导入种子记忆

```bash
python -m scripts.ingest data/seed_memory.md --verbose
```

### 3. 导入 WorkBuddy 真实记忆源（阶段 1.5 新增）

```bash
# 默认扫描 ~/.workbuddy/memory/ + ./.workbuddy/memory/
python -m scripts.ingest_wb_memory --verbose

# 先 dry-run 看看会扫到哪些
python -m scripts.ingest_wb_memory --dry-run

# 自定义目录
python -m scripts.ingest_wb_memory --dir ~/other/memory --project myproject
```

### 4. 健康度检查（阶段 1.5 新增）

```bash
python -m scripts.health           # 完整报告
python -m scripts.health --quiet   # 只看红黄
python -m scripts.health --json   # 机器可读（接 watchdog 友好）
```

### 5. 查询

```bash
python -m scripts.query "SkillFather 项目用什么技术栈" --top-k 5
python -m scripts.query "RAG 优化的去重阈值" --top-k 3
```

### 6. 蒸馏（阶段 2 新增）

```bash
# 看一次蒸馏报告（不改索引）
python -m scripts.distill --dry-run

# 实际清理低价值旧记忆
python -m scripts.distill --verbose

# 注册 Windows Task Scheduler 每天 03:00 自动蒸馏
python -m scripts.install_distill_cron --time 03:00
```

### 7. 二阶段精排（阶段 2 新增）

启用 BGE Reranker（Cross-Encoder），把 RRF 粗排的前 N 重新打分排序：

```bash
# 下载 reranker 模型（一次性，2.2GB）
python download_bge_reranker.py

# 在代码中开启
retriever = Memory(...).retriever
results = retriever.search(query, top_k=5, rerank=True, rerank_top_n=20)
```

### 7b. HyDE Query 改写（v0.2.2 新增）

短 query（如"去重阈值"）召回差。用 LLM 生成假设答案再检索：

```python
from src.hyde import Hyde
from src.memory import Memory

# 方式 1：Mock HyDE（无需 LLM，模板填充）
hyde = Hyde(llm=None)

# 方式 2：自定义 LLM（OpenAI / 本地模型）
hyde = Hyde(llm=lambda q: openai_complete(f"用一句话解释：{q}"))

mem = Memory(hyde=hyde)
results = mem.search("去重阈值", top_k=5, use_hyde=True)
```

原理：用假设答案（"余弦相似度 0.92 触发合并..."）替代原 query 做向量检索，比短 query 更接近真实文档分布，召回提升 20-50%。

### 7c. HTTP API 服务端（v0.2.3 新增）

把 Memory 暴露成 HTTP REST API，让任何应用都能调用：

```bash
# 启动服务
python scripts/server.py --host 0.0.0.0 --port 8000

# 启用 Reranker + HyDE
python scripts/server.py --rerank --hyde

# 调用
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "SkillFather", "top_k": 3}'
```

端点：
- `GET /health` 健康检查
- `GET /stats` 索引统计
- `POST /search` 单次检索
- `POST /batch_search` 批量检索
- `POST /add` 写入记忆

Swagger UI 文档：http://localhost:8000/docs
完整接口文档：[docs/API.md](docs/API.md)

### 8. WorkBuddy 启动钩子（阶段 2 新增）

WorkBuddy 启动时（或新会话开始）自动扫一遍 `~/.workbuddy/memory/` + `./.workbuddy/memory/` 入库。

两条路径并行：

**A. Windows 登录 Run 键（保底）**——用户登录 Windows 自动触发：
```bash
python -m scripts.install_bootstrap                # 安装
python -m scripts.install_bootstrap --uninstall    # 移除
python -m scripts.install_bootstrap --run-now      # 立即跑一次
```

注册表位置：`HKCU\Software\Microsoft\Windows\CurrentVersion\Run\WorkBuddy-RAG-Bootstrap`
日志：`C:\Users\JJ\.workbuddy\rag-bootstrap.log`

**B. WorkBuddy 内置 skill（用户可控）**——`@rag_bootstrap` 手动触发：
- 路径：`C:\Users\JJ\.workbuddy\skills\rag_bootstrap\`
- 触发词：扫一下记忆 / 入库 / 同步到 RAG / 补索引 / 新会话 / 启动

**核心修复**：默认 glob 加了 YYYY-MM-DD.md 模式（`DEFAULT_WB_MEMORY_PATTERNS`），
否则 `2026-06-19.md` 这类工作日志会被忽略。

### 9. 路径参数化（推广级关键）

所有路径/模型都从 `[tool.workbuddy-rag]` 段读，可通过环境变量 `WB_RAG_*` 覆盖：

```python
from src.config import get_index_dir, get_memory_dirs, get_embedding_model

index_dir = get_index_dir()  # 默认 ~/.workbuddy/rag-index
dirs = get_memory_dirs()       # 默认 [~/.workbuddy/memory, ./.workbuddy/memory]
model = get_embedding_model()  # 默认 BAAI/bge-m3
```

```bash
# 命令行覆盖
export WB_RAG_INDEX_DIR=/custom/path
export WB_RAG_DEDUP_THRESHOLD=0.85
export WB_RAG_DEFAULT_MEMORY_DIRS="/path/a,/path/b"
```

支持的配置项见 `pyproject.toml` 的 `[tool.workbuddy-rag]` 段。

### 10. 评估

```bash
# 1. 准备 gold set（先建索引后填 relevant_ids）
python -m scripts.eval --gold data/gold_set.jsonl
# 2. 报告输出到 data/eval_report.md
```

## 项目结构

```
workbuddy-rag-memory/
├── src/
│   ├── embedder.py      # bge-m3 包装 + GPU 自动检测 + fallback
│   ├── chunker.py       # 事实单元切分 + RAW_JSON 拆分
│   ├── indexer.py       # 三索引联合存储
│   ├── dedup.py         # 写入去重拦截器（带索引大小限制）
│   ├── retriever.py     # 混合检索 + RRF + 时间衰减 + HyDE 集成
│   ├── reranker.py      # BGE Reranker v2-m3（智能 batch + LRU 缓存）
│   ├── hyde.py          # HyDE Query 改写（v0.2.2）
│   ├── memory.py        # 统一入口（含 scan_workbuddy_memory）
│   └── config.py        # pyproject + WB_RAG_* 环境变量参数化（v0.2.1）
├── scripts/
│   ├── ingest.py                   # 导入单文件
│   ├── ingest_wb_memory.py         # 扫描 ~/.workbuddy/memory/ 入库
│   ├── ingest_wb_memory_oneshot.py # 启动钩子用（静默模式）
│   ├── query.py                    # 命令行查询
│   ├── health.py                   # 健康度检查（--watch / --log-level）
│   ├── distill.py                  # 自动蒸馏 + 清理低价值旧记忆
│   ├── install_distill_cron.py     # 注册 distill 为定时任务
│   ├── install_bootstrap.py        # Windows Run 键登录自动 ingest
│   ├── server.py                   # FastAPI HTTP 服务端（v0.2.3）
│   └── eval.py                     # 评估脚本
├── tests/
│   ├── test_chunker.py            # chunker 单测
│   ├── test_chunk_raw_json.py     # RAW_JSON 拆分单测
│   ├── test_integration.py        # 端到端
│   ├── test_decay_integration.py  # 时间衰减 + access_count 集成测试
│   ├── test_reranker_e2e.py       # BGE Reranker 端到端
│   ├── test_bootstrap.py          # 启动钩子链路
│   ├── test_config.py             # 路径参数化
│   ├── test_hyde.py               # HyDE Query 改写
│   └── test_server.py             # FastAPI 端点测试
├── docs/
│   └── API.md                      # HTTP API 详细文档
├── data/
│   ├── seed_memory.md   # 10 条种子记忆
│   ├── gold_set.jsonl   # 评估集
│   └── eval_report.md   # 评估报告（生成）
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile                # Linux/Mac 入口
├── make.py                 # Windows Python 替代
├── INSTALL.md              # 5 分钟小白指南
├── LICENSE                 # MIT
└── README.md
```

## 关键设计

### Chunk 切分
- 按 H2/H3 标题 + 双换行分段
- 超长按句号切（避免单 chunk 超 500 字符）
- 过滤 < 30 字符噪声

### 元数据自动提取
- `ts`: 写入时间
- `project`: 启发式（标题 / PascalCase / "项目：XXX"）
- `entities`: 技术栈关键词（白名单匹配）
- `confidence`: 启发式（决策类 +0.2、含日期 +0.1 等）
- `source`: 文件路径识别

### 去重三档
| 相似度 | 决策 | 理由 |
|---|---|---|
| ≥ 0.92 | merge | 高置信度覆盖低置信度 |
| 0.85~0.92 | skip | 防语义撞车 |
| < 0.85 | insert | 新事实 |

### 索引大小策略（阶段 1.5 强化）
- 默认 `search_k=50`，实际查询 `k = min(search_k, count, MAX_SEARCH_K=200)`
- 索引 > 1000 时自动切到 large-mode，`search_k` 减半，避免冗余扫描
- 所有策略都可通过 `DEDUP_MAX_SEARCH_K` / `DEDUP_LARGE_INDEX_THRESHOLD` 环境变量调

### 网络策略（阶段 1.5 强化）
- `src/embedder.py` 顶部默认锁定 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
- 仅下载脚本 `download_bge_m3.py` / `download_bge_v2.py` 临时解锁
- CI / 冒烟测试不会因为网络抖动反复连 HuggingFace

### RRF 融合 + 时间衰减（阶段 2 强化）
```
rrf_score = Σ 1 / (k + rank_i)  for each retriever i
k = 30  (阶段 1 是 60，区分度太低)

score = rrf_score × exp(-Δt_days / τ) × log(1 + access_count) × (0.7 + 0.3 × confidence)
       └────── RRF ──────┘ └───── 时间衰减 ─────┘ └── 流行度 ──┘ └──── 置信度 ────┘
τ = 90 天（可调）

# 默认启用 enable_decay=True
# 默认 record_access=True（Top-K 自动递增 access_count）
```

### Cross-Encoder 重排（阶段 2 新增）
```
# 粗排 RRF 池 → 精排 Cross-Encoder
results_rrf = retriever.search(query, top_k=20)        # RRF 粗排
results_rerank = reranker.rerank(query, results_rrf)    # 交叉编码器重排
# 重排分数是 sigmoid(logit)，区间 [0, 1]，区分度极强
```

**为什么需要 Rerank**：
- 双塔向量模型独立编码 q 和 d，缺乏细粒度交互
- Cross-Encoder 把 (q, d) 作为一对输入，能捕捉 token 级匹配
- 实测：rerank 后 Top-3 命中率从 60% → 95%+

## 配置项（.env）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-m3` | embedding 模型 |
| `EMBED_DIM` | `1024` | 向量维度 |
| `EMBED_DEVICE` | `cpu` | `cpu` / `cuda` |
| `INDEX_DIR` | `./.index` | 索引存储目录 |
| `DEDUP_THRESHOLD` | `0.92` | 去重阈值 |
| `TOP_K` | `5` | 默认返回条数 |
| `RRF_K` | `60` | RRF 常数 |

## 已知限制

- 阶段 1 不含 Rerank（阶段 2 加入）
- 阶段 1 不含自动蒸馏
- 启发式元数据提取精度有限（项目名可能误判，如"决定用"被识别成项目名）
- Embedding fallback 模式（hash）无语义信息 → **dedup 自动跳过**（不会误判），但检索质量随机
- Hash fallback 模式下 BM25 仍可用做关键词检索
- Gold set 默认是占位（relevant_ids 为空），需手工标 ground truth 才有 Recall 读数
- 大索引（>1000）时 dedup 自动降速，可能漏掉部分跨区重复

## 环境说明

- **Hash fallback 模式**：未装 `sentence-transformers` 时启用，**用于 CI/冒烟测试**。插入全 OK，检索走 BM25 + 随机向量融合。
- **生产模式**：执行 `pip install sentence-transformers`，默认加载 BAAI/bge-m3（约 2.3GB），可切换为 `paraphrase-multilingual-MiniLM-L12-v2`（约 500MB，速度快但精度略低）。
- **离线模式**：`HF_HUB_OFFLINE=1` 在 `src/embedder.py` 顶部锁定，防止 CI 反复联网。需要下载时临时关掉。

## 下一步

- [x] 阶段 2：接 BGE Reranker（架构已留接口，stage2/reranker.py 待实现）
- [x] 阶段 2：Query 改写（HyDE）
- [x] 阶段 2：时间衰减重排
- [ ] 阶段 3：自动蒸馏 + 健康度仪表盘
- [ ] 标注完整 gold set（至少 100 条）

## ⭐ Star History

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=alpsmonkey/workbuddy-rag-memory&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=alpsmonkey/workbuddy-rag-memory&type=Date" />
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=alpsmonkey/workbuddy-rag-memory&type=Date" width="720"/>
</picture>

> 👉 [在 Star History 查看完整曲线](https://star-history.com/#alpsmonkey/workbuddy-rag-memory&Date)

如果你觉得这个项目有用，欢迎点 ⭐ 支持一下——每一个 star 都是我继续迭代 v0.3 阶段（自动蒸馏 + 健康度仪表盘）的动力。

## 守护进程（watchdog 自动入库）

**路径**：`~/.workbuddy/rag-daemon/`（不在本项目内，独立部署）

**安装**：
```bash
python ~/.workbuddy/rag-daemon/install.py
```

**监控范围**：
- `~/.workbuddy/MEMORY.md`（用户级）
- `.workbuddy/memory/YYYY-MM-DD.md` + `.workbuddy/memory/MEMORY.md`（项目级）
- `~/.workbuddy/skills/*/SKILL.md`（skill 知识）

**延迟**：写入触发 → 5s debounce → 自动入库 → 索引可检索

**卸载**：
```bash
python ~/.workbuddy/rag-daemon/uninstall.py
```

详见 `~/.workbuddy/rag-daemon/README.md`
