# WorkBuddy RAG 增强记忆系统 - 阶段 2

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
│   ├── embedder.py      # bge-m3 包装 + fallback（默认 HF 离线）
│   ├── chunker.py       # 事实单元切分 + 元数据
│   ├── indexer.py       # 三索引联合存储
│   ├── dedup.py         # 写入去重拦截器（带索引大小限制）
│   ├── retriever.py     # 混合检索 + RRF + 时间衰减 + access_count
│   ├── reranker.py      # BGE Reranker v2-m3 包装（Cross-Encoder）
│   └── memory.py        # 统一入口（含 scan_workbuddy_memory）
├── scripts/
│   ├── ingest.py                   # 导入单文件
│   ├── ingest_wb_memory.py         # 扫描 ~/.workbuddy/memory/ 入库（阶段 1.5 新增）
│   ├── query.py                    # 命令行查询
│   ├── health.py                   # 健康度检查（阶段 1.5 新增）
│   ├── distill.py                  # 自动蒸馏 + 清理低价值旧记忆（阶段 2 新增）
│   ├── install_distill_cron.py     # 注册 distill 为定时任务（阶段 2 新增）
│   └── eval.py                     # 评估脚本
├── tests/
│   ├── test_chunker.py            # chunker 单测
│   ├── test_chunk_raw_json.py     # RAW_JSON 拆分单测
│   ├── test_integration.py        # 端到端
│   ├── test_decay_integration.py  # 时间衰减 + access_count 集成测试
│   ├── test_reranker_e2e.py       # BGE Reranker 端到端（阶段 2 新增）
│   └── test_bootstrap.py          # 启动钩子链路（阶段 2 新增）
├── data/
│   ├── seed_memory.md   # 10 条种子记忆
│   ├── gold_set.jsonl   # 评估集
│   └── eval_report.md   # 评估报告（生成）
├── requirements.txt
├── pyproject.toml
└── .env.example
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
