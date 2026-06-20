# WorkBuddy RAG Memory HTTP API 文档

> v0.2.3 新增：通过 HTTP 调用检索 / 入库能力

## 启动

```bash
# 默认（localhost:8000）
python scripts/server.py

# 公网访问
python scripts/server.py --host 0.0.0.0 --port 8000

# 启用 Reranker + HyDE
python scripts/server.py --rerank --hyde

# 开发模式（自动重载）
python scripts/server.py --reload
```

启动后访问：
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc
- OpenAPI JSON：http://localhost:8000/openapi.json

## 端点

### GET `/health`

健康检查。

**响应**：
```json
{
  "status": "ok",
  "version": "0.2.3",
  "index_dir": "C:\\Users\\<you>\\.workbuddy\\rag-index",
  "backend": "sentence-transformers"
}
```

### GET `/stats`

索引统计。

**响应**：
```json
{
  "total_chunks": 159,
  "total_projects": 16,
  "backend": "sentence-transformers",
  "embedding_dim": 1024,
  "index_dir": "..."
}
```

### POST `/search`

检索。

**请求体**：
```json
{
  "query": "SkillFather 用了什么技术栈",
  "top_k": 5,
  "project": null,
  "source": null,
  "rerank": false,
  "use_hyde": false,
  "candidates": 20
}
```

**响应**：
```json
{
  "query": "SkillFather 用了什么技术栈",
  "hits": [
    {
      "id": "abc123",
      "text": "SkillFather 是 Python 项目...",
      "score": 0.8765,
      "project": "SkillFather",
      "source": "user-memory",
      "ts": "2026-06-20T10:00:00",
      "confidence": 0.8,
      "rerank_score": null,
      "access_count": 5
    }
  ],
  "count": 5,
  "duration_ms": 23.5
}
```

### POST `/batch_search`

批量检索（一次请求多个 query）。

**请求体**：
```json
{
  "queries": ["SkillFather", "去重阈值", "时间衰减"],
  "top_k": 3,
  "rerank": false,
  "use_hyde": false
}
```

**响应**：
```json
{
  "results": [
    {"query": "SkillFather", "hits": [...], "count": 3, "duration_ms": 23},
    {"query": "去重阈值", "hits": [...], "count": 2, "duration_ms": 18},
    {"query": "时间衰减", "hits": [...], "count": 3, "duration_ms": 21}
  ]
}
```

### POST `/add`

写入单条记忆。

**请求体**：
```json
{
  "text": "## 决策\nRAG 用 bge-m3 + LanceDB",
  "source": "api-upload",
  "project": "MyProject"
}
```

**响应**：
```json
{
  "decision": "insert",
  "reason": "new chunk",
  "existing_id": null,
  "similarity": 0.0
}
```

`decision` 取值：
- `insert`：新写入
- `merge`：与已存在 chunk 合并（高相似度）
- `skip`：重复，跳过（中相似度）

## 客户端调用示例

### cURL

```bash
# 检索
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "SkillFather", "top_k": 3}'

# 入库
curl -X POST http://localhost:8000/add \
  -H "Content-Type: application/json" \
  -d '{"text": "## 决定\n用 Python 3.13"}'

# 健康检查
curl http://localhost:8000/health
```

### Python (requests)

```python
import requests

BASE = "http://localhost:8000"

# 检索
r = requests.post(f"{BASE}/search", json={
    "query": "SkillFather",
    "top_k": 5,
    "rerank": True,
})
for hit in r.json()["hits"]:
    print(f"[{hit['score']:.3f}] {hit['text'][:80]}")

# 批量检索
r = requests.post(f"{BASE}/batch_search", json={
    "queries": ["A", "B", "C"],
    "top_k": 3,
})
for result in r.json()["results"]:
    print(f"{result['query']}: {result['count']} hits")
```

### JavaScript (fetch)

```javascript
const res = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'SkillFather', top_k: 5 })
});
const data = await res.json();
console.log(data.hits);
```

## 部署

### Docker

```bash
docker build -t workbuddy-rag-memory .
docker run -p 8000:8000 \
  -v ~/.workbuddy:/root/.workbuddy \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  workbuddy-rag-memory python scripts/server.py --host 0.0.0.0
```

### 配合 nginx 反向代理

```nginx
location /rag/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## 限流与鉴权

v0.2.3 暂未实现，建议：
- 用 nginx limit_req 做 IP 限流
- 生产部署加 OAuth2 / API Key 校验（中间件层）
- 不要直接暴露在公网，加防火墙

## 性能参考

| 操作 | 延迟（CPU） | 延迟（GPU） |
|---|---|---|
| `/search`（5 results，无 rerank） | 30-80ms | 20-50ms |
| `/search`（带 rerank） | 1-3s | 200-500ms |
| `/batch_search`（10 queries） | 300-800ms | 200-500ms |
| `/add` | 10-50ms | 10-50ms |

## 错误码

| 状态码 | 含义 |
|---|---|
| 200 | 成功 |
| 422 | 请求参数错误（pydantic 校验失败） |
| 500 | 服务端错误（Memory 操作失败） |
| 503 | 服务未就绪（启动中） |