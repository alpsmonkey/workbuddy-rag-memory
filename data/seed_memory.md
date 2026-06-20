# 鹏哥偏好: 用 Python 3.13, 习惯 5 个/批节奏提问, 偏好表格化展示, 显式范围校验

# SkillFather 项目

## 项目背景
SkillFather 是 WorkBuddy 跨平台 Agent Skill 适配度分析工具，v0.2.1 版本。
从使用角度分析一个 Agent Skill 是否适用，基于 README 自动生成 6-10 个诊断问题。
输出 5 维度适配度评分（满分 10 分）。

## 技术决策
2026-05-28 决定 SkillFather 用 Python 开发，核心是 5 维评分引擎。
弃用 GraphQL 作为 API 层，理由是复杂度高于收益，团队 REST 已够用。

## 评分维度
SkillFather 5 个评分维度分别是：可发现性、可执行性、可维护性、依赖清晰度、文档质量。
每个维度满分 10 分，最终加权平均。

# WorkBuddy 记忆架构

## 现有架构
WorkBuddy 记忆分 3 层：Layer 1 云端记忆（auto-injected profile + conversation_search）。
Layer 2 用户级本地记忆 ~/.workbuddy/MEMORY.md，限制 4,000 字符。
Layer 3 项目工作区记忆 .workbuddy/memory/ 目录，每天日志 + MEMORY.md，限制 3,000 字符。

## 已知问题
WorkBuddy 记忆架构有 8 个真实缺点。
最大问题是写入完全靠自律，没有冲突检测，字符硬上限太死。
另一个问题是缺乏语义检索，跨项目历史只能 grep 或 conversation_search。

# RAG 增强记忆优化

## 优化目标
RAG 增强记忆优化分 3 个阶段实施。
阶段 1 目标：chunk 切分 + LanceDB + FTS5 双索引 + 去重 + 评估。
阶段 2 目标：BGE Rerank + Query 改写 + 时间衰减。
阶段 3 目标：自动蒸馏 + 健康度仪表盘。

## 技术选型
embedding 模型用 BAAI/bge-m3，1024 维中英双语。
向量库选 LanceDB 0.10+ 嵌入式部署。
关键词检索用 SQLite FTS5 引擎。
三路融合用 RRF 倒数排序融合算法。

## 去重策略
chunk 写入前算 embedding，查最近 100 条最相似。
余弦相似度阈值 0.92 触发合并决策。
相似度在 0.85 到 0.92 之间跳过，防语义撞车。
低于 0.85 直接 insert 作为新事实。

## 时间衰减
时间衰减公式 score = sim × exp(-Δt_days / 90) × log(1+访问次数)。
τ 常数取 90 天，访问次数 log 缩放防热门记忆霸榜。
