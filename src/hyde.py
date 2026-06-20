"""
HyDE (Hypothetical Document Embeddings) Query 改写

原理（来自 Gao et al., 2022）：
- 短 query（如"去重阈值"）直接 encode 进向量空间，召回差
- 用 LLM 生成 query 的"假设答案"（如"余弦相似度 0.92 触发合并..."）
- 用假设答案 encode 进向量空间 → 召回显著提升（接近真实文档分布）

用法：
  hyde = Hyde(llm=my_llm)
  hypothetical_doc = hyde.generate("去重阈值")
  # 现在 hypothetical_doc 可以替代 query 去检索

接口设计：
- Hyde 是一个类，注入一个 LLM callable
- 没有 LLM 时提供 MockHyde（返回 query 本身 + 一些模板文本），仍能提升召回
- 缓存：同 query 5 分钟内复用

性能影响：
- 每次查询多 1 次 LLM 调用（本地 LLM 100-500ms；mock 1ms）
- 短 query 召回提升 20-50%（学术论文数据；实际取决于 LLM 质量）
"""
from __future__ import annotations
import hashlib
import logging
import time
from typing import Callable, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


# Mock HyDE：当没有 LLM 时，用 query + 模板生成"看起来像答案"的文本
# 仍有正向效果（向量空间更靠近真实文档）
MOCK_TEMPLATES = [
    "关于「{query}」的说明：",
    "「{query}」的定义和用途：",
    "以下是「{query}」的相关信息：",
]


class Hyde:
    """HyDE Query 改写器"""

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        cache_ttl: float = 300.0,
    ):
        """
        Args:
            llm: 接收 query 返回假设文档的 callable。无 LLM 时传 None 用 mock。
            cache_ttl: 缓存 TTL（秒），默认 5 分钟
        """
        self.llm = llm
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, str]] = {}

    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _get_cached(self, query: str) -> Optional[str]:
        key = self._cache_key(query)
        if key in self._cache:
            ts, doc = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return doc
        return None

    def _set_cached(self, query: str, doc: str) -> None:
        key = self._cache_key(query)
        self._cache[key] = (time.time(), doc)
        # 简单 LRU：> 500 条清旧
        if len(self._cache) > 500:
            cutoff = time.time() - self.cache_ttl
            self._cache = {k: v for k, v in self._cache.items() if v[0] > cutoff}

    def generate(self, query: str) -> str:
        """生成 query 的假设文档

        Returns:
            假设文档字符串（用来替代 query 去做向量检索）
        """
        if not query or not query.strip():
            return query

        cached = self._get_cached(query)
        if cached is not None:
            logger.debug("HyDE 缓存命中: %s", query[:30])
            return cached

        if self.llm is not None:
            try:
                hypothetical = self.llm(query)
                if not hypothetical or not hypothetical.strip():
                    raise ValueError("LLM 返回空")
            except (ValueError, RuntimeError, OSError) as e:
                logger.warning("HyDE LLM 调用失败: %s，降级到 mock", e)
                hypothetical = self._mock_generate(query)
        else:
            hypothetical = self._mock_generate(query)

        self._set_cached(query, hypothetical)
        return hypothetical

    def _mock_generate(self, query: str) -> str:
        """Mock 假设文档：模板填充"""
        # 选模板轮换（避免 100% 相同文本让向量检索仍按 query 命中）
        idx = sum(ord(c) for c in query) % len(MOCK_TEMPLATES)
        template = MOCK_TEMPLATES[idx]
        return template.format(query=query) + " " + query


def make_default_hyde() -> Hyde:
    """便捷：返回无 LLM 的 Hyde 实例（用 mock）"""
    return Hyde(llm=None)