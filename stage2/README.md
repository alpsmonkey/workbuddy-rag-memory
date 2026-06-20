# 阶段 2：检索质量增强

## 目标

在阶段 1（三索引 + 去重 + RRF）基础上，提升检索精度。

## 候选方向

### 1. BGE Reranker 重排
- 模型：`BAAI/bge-reranker-v2-m3`（与 bge-m3 配套）
- 作用：对 Top-20 RRF 结果重排 → Top-5
- 预期：Recall@5 +10~15%
- 状态：架构留接口（`Retriever.rerank()`），待实现

### 2. HyDE Query 改写
- 思路：用 LLM 生成"假设答案"作为检索 query
- 适用：抽象问题（"怎么优化 RAG？"）
- 状态：未实现

### 3. 时间衰减重排
- 公式：`score = sim × exp(-Δt_days / 90) × log(1 + access_count)`
- 作用：老旧记忆自然下沉，高频访问记忆上浮
- 状态：公式已定义在 `seed_memory.md`，待接入 retriever

## 验收指标

| 指标 | 阶段 1 基线 | 阶段 2 目标 |
|---|---|---|
| Recall@5 | 0.80 | ≥ 0.90 |
| Recall@10 | 0.80 | ≥ 0.95 |
| MRR | 0.725 | ≥ 0.85 |
| 延迟 P50 | 200ms | ≤ 250ms（含 rerank） |

## 实施顺序

1. 时间衰减（最容易，1-2 天）
2. BGE Reranker（需要装新模型，半天）
3. HyDE（需要 LLM，1-2 天）