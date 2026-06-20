"""
事实单元切分 + 元数据提取
- 按 H2/H3 + 段落语义切分
- 启发式提取: 项目/技术栈/时间/置信度/来源
- 特殊处理: RAW_JSON 块 / 表格块 单独切分
"""
from __future__ import annotations
import re
import json
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# 常见技术栈/项目关键词（启发式，可扩展）
TECH_KEYWORDS = {
    "python", "javascript", "typescript", "react", "vue", "vite", "webpack",
    "node", "nodejs", "lancedb", "sqlite", "postgres", "mysql",
    "fastapi", "flask", "django", "express", "nextjs", "nuxt",
    "bge-m3", "embedding", "rerank", "rag", "lancedb", "fts5",
    "sap", "fico", "co", "rfc", "abap",
    "remotion", "imagegen", "skill", "skillfather", "workbuddy",
    "windows", "linux", "macos", "wsl", "git", "github",
}

# 关键术语白名单（短但必须保留的术语）
KEY_TERMS = {
    "rag", "rrf", "bm25", "fts5", "lancedb", "sqlite",
    "bge-m3", "embedding", "rerank", "reranker",
    "hyde", "query rewrite", "时间衰减", "去重",
    "向量检索", "语义检索", "关键词检索", "混合检索",
    "元数据", "融合", "嵌入", "向量",
    "lance", "ann", "hnsw", "ivf",
    "watchdog", "daemon", "ingest", "dedup", "retriever",
}

# 项目名启发式
# 优先级：模式 1（"项目: XXX"） > 模式 2（PascalCase）
# 注意：避免 "项目背景"、"技术决策" 等 H2 通用词被误识别
PROJECT_PATTERNS = [
    # 模式 1：必须 "项目" + 冒号/空格 + 名字（避免 "项目背景" 这种误匹配）
    re.compile(r"项目[:：\s]+([A-Za-z0-9\-_一-龥]{2,20})"),
    # 模式 2：PascalCase / dotted（必须 4+ 字符，避免 RAG/API/BGE 这种短缩略词）
    re.compile(r"\b([A-Z][A-Za-z0-9]{3,15}(?:\.[a-z]+)?)\b"),
]

# H2 / 通用词黑名单（绝不当 project）
GENERIC_H2_BLACKLIST = {
    "项目背景", "技术决策", "评分维度", "现有架构", "已知问题",
    "优化目标", "技术选型", "去重策略", "时间衰减", "快速开始",
    "项目结构", "关键设计", "配置项", "已知限制", "环境说明",
    "下一步", "守护进程", "完整产物清单",
    "测试", "结果", "结论", "建议", "问题", "回答",
    "背景", "决策", "架构", "问题", "目标", "选型",
    "summary", "background", "requirements", "design",
    "implementation", "testing", "results", "discussion",
}

# RAW_JSON 块标记（WorkBuddy conversation_search 注入的元数据块）
RAW_JSON_START = "<!-- RAW_JSON_START -->"
RAW_JSON_END = "<!-- RAW_JSON_END -->"
RAW_JSON_PATTERN = re.compile(
    re.escape(RAW_JSON_START) + r"\s*\n(.*?)\n\s*" + re.escape(RAW_JSON_END),
    re.DOTALL,
)


class Chunk(BaseModel):
    """事实单元"""
    id: str = Field(default_factory=lambda: f"chunk_{datetime.now().timestamp():.0f}_{hash_id()}")
    text: str
    meta: Dict = Field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {"id": self.id, "text": self.text, **self.meta}


def hash_id() -> str:
    import random
    return f"{random.randint(0, 99999):05d}"


def is_key_term(text: str) -> bool:
    """检测文本是否含关键术语"""
    if not text:
        return False
    text_lower = text.lower()
    for term in KEY_TERMS:
        if term in text_lower:
            return True
    return False


def extract_project(text: str) -> Optional[str]:
    """启发式提取项目名"""
    for pat in PROJECT_PATTERNS:
        m = pat.search(text)
        if m:
            name = m.group(1)
            # 过滤 H2 通用词
            if name in GENERIC_H2_BLACKLIST:
                continue
            return name
    return None


def extract_entities(text: str) -> List[str]:
    """提取技术栈实体"""
    text_lower = text.lower()
    found = []
    for kw in TECH_KEYWORDS:
        if kw in text_lower and kw not in found:
            found.append(kw)
    return found[:10]


def detect_source(text: str, source_path: str = None) -> str:
    """检测记忆来源"""
    if source_path:
        if "MEMORY.md" in source_path:
            return "user-memory"
        if ".workbuddy/memory" in source_path:
            return "workspace-log"
        if "conversation" in source_path.lower():
            return "conversation"
    if text.startswith("#") or text.startswith("##"):
        return "user-memory"
    return "unknown"


def compute_confidence(text: str) -> float:
    """启发式置信度"""
    score = 0.5
    if len(text) > 50:
        score += 0.1
    if any(kw in text.lower() for kw in ["决定", "弃用", "采用", "用 ", "不用", "失败", "成功"]):
        score += 0.2
    if re.search(r"\d{4}-\d{2}-\d{2}", text):
        score += 0.1
    if re.search(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+", text):
        score += 0.1
    return min(score, 1.0)


def _make_chunk(text: str, source: str = None, meta_overrides: Dict = None) -> Chunk:
    meta = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "project": extract_project(text),
        "entities": extract_entities(text),
        "confidence": compute_confidence(text),
        "source": detect_source(text, source),
        "length": len(text),
    }
    if meta_overrides:
        meta.update(meta_overrides)
    return Chunk(text=text, meta=meta)


def _split_raw_json_block(text: str, source: str, min_length: int) -> List[Chunk]:
    """
    专门处理 RAW_JSON 块：按 key 拆成多个 chunk，而不是整块一个

    原始结构（注入的元数据）：
    {
      "memory_type": "user_preference",
      "summary": "...",
      "key_facts": ["...", "..."],
      "tags": ["..."]
    }

    拆分成：
    - chunk: "memory_type: user_preference"
    - chunk: "summary: ..."
    - chunk: "key_facts: ..."（多行合并一行）
    - chunk: "tags: ..."
    """
    chunks = []
    text_stripped = text.strip()

    try:
        data = json.loads(text_stripped)
    except (json.JSONDecodeError, ValueError):
        if len(text_stripped) >= min_length or is_key_term(text_stripped):
            chunks.append(_make_chunk(text_stripped, source, {"block_type": "raw_json"}))
        return chunks

    if not isinstance(data, dict):
        # 顶层不是对象（如数组），退化为文本
        line = json.dumps(data, ensure_ascii=False)
        if len(line) >= min_length or is_key_term(line):
            chunks.append(_make_chunk(line, source, {"block_type": "raw_json"}))
        return chunks

    # 关键字段优先提取
    priority_keys = [
        "memory_type", "summary", "key_facts", "facts", "decisions",
        "preferences", "tags", "topics",
    ]
    memory_type = data.get("memory_type", "unknown")

    for k in priority_keys:
        if k not in data:
            continue
        v = data[k]
        if isinstance(v, list):
            joined = "; ".join(str(x) for x in v)
            line = f"{k}: {joined}"
        elif isinstance(v, dict):
            line = f"{k}: " + "; ".join(f"{kk}={vv}" for kk, vv in v.items())
        else:
            line = f"{k}: {v}"

        if len(line) >= min_length or is_key_term(line):
            chunks.append(_make_chunk(line, source, {
                "block_type": "raw_json",
                "json_key": k,
                "memory_type": memory_type,
            }))

    # 其他字段也尽量保留
    for k, v in data.items():
        if k in priority_keys:
            continue
        if isinstance(v, (str, int, float, bool)):
            line = f"{k}: {v}"
            if len(line) >= min_length or is_key_term(line):
                chunks.append(_make_chunk(line, source, {
                    "block_type": "raw_json",
                    "json_key": k,
                    "memory_type": memory_type,
                }))

    return chunks


def chunk_text(
    text: str,
    source: str = None,
    min_length: int = 8,
    max_length: int = 500,
) -> List[Chunk]:
    """
    按"事实单元"切分:
    - Step 0: 抽取 RAW_JSON 块，单独切分（按 key 拆开）
    - Step 1: 按 ## / ### 标题分大段
    - Step 2: 大段内再按双换行分段
    - Step 3: 单段超 max_length 时按句号切
    - Step 4: 过滤太短噪声，但 KEY_TERMS 白名单豁免
    """
    if not text or not text.strip():
        return []

    chunks: List[Chunk] = []

    # Step 0: 先把 RAW_JSON 块抽出来，剩余部分走普通切分
    remaining = text
    raw_json_blocks = list(RAW_JSON_PATTERN.finditer(text))
    if raw_json_blocks:
        for m in raw_json_blocks:
            json_block = m.group(1)
            json_chunks = _split_raw_json_block(json_block, source, min_length)
            chunks.extend(json_chunks)
        # 从 remaining 中移除 RAW_JSON 块（防止重复切分）
        # 用 | 拆分 + filter 重建，避免 replace 顺序问题
        parts = []
        last_end = 0
        for m in raw_json_blocks:
            parts.append(text[last_end:m.start()])
            parts.append(" ")  # 占位
            last_end = m.end()
        parts.append(text[last_end:])
        remaining = "".join(parts)

    # Step 1: 按 H2/H3 切
    sections = re.split(r"\n(?=#{1,3}\s)", remaining.strip())

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Step 2: 段内按双换行切
        paragraphs = [p.strip() for p in re.split(r"\n\n+", section) if p.strip()]

        for para in paragraphs:
            # Step 3: 超长按句号切
            if len(para) > max_length:
                sentences = re.split(r"([。！？；\.\!\?\;])", para)
                buf = ""
                for s in sentences:
                    buf += s
                    if len(buf) > max_length // 2 and re.search(r"[。！？；\.\!\?\;]$", s):
                        stripped = buf.strip()
                        if len(stripped) >= min_length or is_key_term(stripped):
                            chunks.append(_make_chunk(stripped, source))
                        buf = ""
                stripped = buf.strip()
                if stripped and (len(stripped) >= min_length or is_key_term(stripped)):
                    chunks.append(_make_chunk(stripped, source))
            else:
                stripped = para.strip()
                if stripped and (len(stripped) >= min_length or is_key_term(stripped)):
                    chunks.append(_make_chunk(stripped, source))

    return chunks