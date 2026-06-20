# WorkBuddy RAG 增强记忆系统 - 5 分钟安装指南

> 给**小白用户**写的版本。开发者请看 [README.md](README.md)。

## 5 分钟快速跑起来

### 1. 系统要求

- **Python 3.10+**（推荐 3.13）
- **磁盘空间**：8 GB（bge-m3 4.3GB + bge-reranker-v2-m3 2.2GB + 索引）
- **内存**：4 GB（bge-m3 模型加载需要）
- **可选 GPU**：CUDA 加速（CPU 也可，吞吐慢 5-10 倍）

### 2. 一行命令克隆

```bash
git clone https://github.com/your-org/workbuddy-rag-memory.git
cd workbuddy-rag-memory
```

### 3. 创建 venv + 装依赖

**Windows (PowerShell)：**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Git Bash)：**
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

**macOS / Linux：**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. 下载模型（一次性，约 5-10 分钟）

```bash
make download-models
# 或手动：
python download_bge_m3.py
python download_bge_reranker.py
```

> 💡 模型缓存到 `~/.cache/huggingface/`，下次换机器可直接复制。

### 5. 健康度检查

```bash
make health
```

应该看到：

```
============================================================
WorkBuddy RAG 健康度报告
============================================================
[1/6] ✅ 后端: sentence-transformers (bge-m3, 1024维)
[2/6] ✅ 索引: 34 chunks, 16 项目
[3/6] ✅ 存储: 113.5 KB
[4/6] ✅ Dedup: threshold=0.92, max_search_k=200
[5/6] ✅ HF 缓存: 4.3 GB (bge-m3)
[6/6] ✅ 真实记忆源: 2 个文件
============================================================
整体: 🟢 HEALTHY
```

### 6. 准备你的记忆

在 `~/.workbuddy/memory/` 创建你的记忆文件：

**Linux/Mac：**
```bash
mkdir -p ~/.workbuddy/memory
cat > ~/.workbuddy/memory/MEMORY.md <<'EOF'
# 长期记忆

## 偏好
- 用 Python 3.13
- 表格 > 段落
- 显式范围校验

## 决策
- 项目用 uv 管理 venv
- 文档用 Markdown
EOF
```

**Windows (PowerShell)：**
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.workbuddy\memory"
# 用编辑器创建 MEMORY.md
```

### 7. 入库

```bash
make ingest
```

应该看到：

```
📁 扫描目录: ['C:\\Users\\<you>\\.workbuddy\\memory', '.workbuddy\\memory']
📊 扫描: 1 个文件
✅ Memory(index_dir=..., total=3, embedder=...)
```

### 8. 测试检索

```bash
make query Q="你的偏好"
```

---

## 常用命令速查

| 命令 | 作用 |
|---|---|
| `make health` | 健康度检查 |
| `make ingest` | 重新入库 |
| `make query Q="..."` | 检索 |
| `make distill` | 清理低价值旧记忆 |
| `make bootstrap` | 安装 Windows 启动钩子（登录自动入库） |
| `make bootstrap-uninstall` | 卸载启动钩子 |
| `make test` | 跑全部 24 个测试 |

---

## 故障排查

### ❌ "ModuleNotFoundError: No module named 'sentence_transformers'"

```bash
# 没激活 venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### ❌ "OSError: [WinError 5] 拒绝访问" (LanceDB Windows)

把项目路径移出 `OneDrive / Dropbox / 网络磁盘`，或确保目录权限开放。

### ❌ 模型下载失败

```bash
# 设置镜像源（中国大陆）
export HF_ENDPOINT=https://hf-mirror.com
python download_bge_m3.py
```

### ❌ 检索无结果

```bash
# 检查健康度
make health
# 看索引里有没有内容
make ingest-dry
```

---

## 下一步

- 看 [README.md](README.md) 了解完整功能
- 编辑 `pyproject.toml` 的 `[tool.workbuddy-rag]` 段自定义路径 / 模型
- 加入 [GitHub Discussions](https://github.com/your-org/workbuddy-rag-memory/discussions) 反馈问题