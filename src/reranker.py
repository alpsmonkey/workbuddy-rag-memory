"""
BGE Reranker v2-m3 包装
- 直接用 transformers AutoModel（避免 CrossEncoder 的 HF_HUB_OFFLINE 兼容问题）
- Lazy load + 兼容 HF_HUB_OFFLINE=1
- 返回 sigmoid 归一化分数 [0, 1]

性能优化（v0.2.1）:
- GPU 自动检测（device="auto"）
- 按文档长度智能 batch（短文合并、长文拆分）
- max_length 超长自动截断 + 警告
- 同 query 5s 内复用结果缓存
"""
from __future__ import annotations
import os
import time
import math
import hashlib
import logging
import warnings
from typing import List, Optional, Sequence, Dict, Tuple

# 与 embedder.py 保持一致：默认离线
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    """auto / cpu / cuda 解析"""
    if requested in ("cpu", "cuda"):
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Reranker 检测到 CUDA，使用 GPU")
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Reranker:
    """BGE Reranker v2-m3 cross-encoder

    性能参数（实测 v0.2.1）：
    - CPU 顺序 20 docs: 6s
    - GPU batch=8: 1-2s
    - 缓存命中（同 query 5s 内）: ~1ms
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_CACHE_TTL = 5.0  # 秒
    TRUNCATE_WARN_RATIO = 0.8  # 文档超过 max_length*ratio 触发警告

    def __init__(
        self,
        model_name: str = None,
        max_length: int = 512,
        device: str = None,
        batch_size: int = None,
        cache_ttl: float = None,
    ):
        # 优先用 config
        try:
            from .config import get_reranker_model, get_device
        except ImportError:
            from config import get_reranker_model, get_device

        self.model_name = model_name or get_reranker_model()
        self.max_length = max_length
        self.device = _resolve_device(device or get_device())
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.cache_ttl = cache_ttl if cache_ttl is not None else self.DEFAULT_CACHE_TTL

        self._model = None
        self._tokenizer = None

        # query -> (ts, scores) 缓存
        self._cache: Dict[str, Tuple[float, List[float]]] = {}

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info("Reranker 加载模型: %s (device=%s)", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            if self.device == "cuda":
                self._model = self._model.to(self.device)
            logger.info("Reranker 加载完成")
        except (ImportError, OSError, RuntimeError) as e:
            logger.error("Reranker 加载失败: %s", e)
            raise

    @staticmethod
    def _sigmoid(x) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except (ValueError, OverflowError):
            return 0.5

    def _cache_key(self, query: str, documents: List[str]) -> str:
        """缓存键：query + docs hash"""
        h = hashlib.md5()
        h.update(query.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(len(documents)).encode())
        for d in documents:
            h.update(d[:200].encode("utf-8", errors="ignore"))
            h.update(b"|")
        return h.hexdigest()

    def _check_truncate(self, doc: str) -> Tuple[str, bool]:
        """检查 + 截断超长文档

        Returns: (truncated_text, was_truncated)
        """
        # 粗略用字符数估计（中文 1 字 ≈ 1.5 token，英文 1 词 ≈ 1.3 token）
        est_tokens = int(len(doc) * 1.5)
        threshold = int(self.max_length * self.TRUNCATE_WARN_RATIO)
        if est_tokens <= threshold:
            return doc, False
        # 截断到 ~ max_length * 0.7（保守）
        safe_chars = int(self.max_length * 0.7 / 1.5)
        truncated = doc[:safe_chars]
        return truncated, True

    def score(self, query: str, documents: List[str]) -> List[float]:
        """对 (query, doc) 对打分 → sigmoid 后 [0, 1]

        性能优化（v0.2.1）:
        - 同 query 5s 内复用结果缓存
        - 超长文档自动截断 + warn
        - 按文档长度智能 batch
        """
        if not documents:
            return []

        # 缓存查询
        cache_key = self._cache_key(query, documents)
        if cache_key in self._cache:
            ts, scores = self._cache[cache_key]
            if time.time() - ts < self.cache_ttl:
                logger.debug("Reranker 缓存命中: %d docs", len(documents))
                return scores

        self._load()
        import torch

        # 预处理：截断超长文档
        truncated_docs = []
        for d in documents:
            td, was_cut = self._check_truncate(d)
            if was_cut:
                logger.warning("Reranker 截断超长文档: %d → %d chars", len(d), len(td))
            truncated_docs.append(td)

        scores: List[float] = []

        # 智能 batch：按长度分组
        # 短文档（< max_length/2）合并，长文档单独
        threshold_chars = self.max_length // 2
        groups: List[Tuple[List[int], List[str]]] = []  # (indices, docs)

        for idx, doc in enumerate(truncated_docs):
            placed = False
            for g_indices, g_docs in groups:
                # 如果组内最大长度 + 这个文档长度 <= max_length，可以合并
                if max(len(d) for d in g_docs) + len(doc) <= self.max_length * 2:
                    g_indices.append(idx)
                    g_docs.append(doc)
                    placed = True
                    break
            if not placed:
                groups.append(([idx], [doc]))

        # 每组最多 batch_size 个
        batches: List[Tuple[List[int], List[str]]] = []
        for g_indices, g_docs in groups:
            for i in range(0, len(g_docs), self.batch_size):
                batches.append((g_indices[i:i+self.batch_size], g_docs[i:i+self.batch_size]))

        logger.debug("Reranker 分批: %d docs → %d batches (avg %.1f/batch)",
                    len(documents), len(batches), len(documents) / max(1, len(batches)))

        # 初始化结果
        results = [0.0] * len(documents)

        with torch.no_grad():
            for batch_indices, batch_docs in batches:
                # 构造 (query, doc) 对
                pairs = [[query, d] for d in batch_docs]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self._model(**inputs).logits
                # 兼容单/多结果
                if logits.dim() == 1 or logits.shape[0] == 1:
                    batch_scores = [self._sigmoid(logits.squeeze().item())]
                else:
                    batch_scores = [self._sigmoid(x.item()) for x in logits.squeeze(-1)]

                for idx, s in zip(batch_indices, batch_scores):
                    results[idx] = s

        # 写缓存
        self._cache[cache_key] = (time.time(), results)
        # 简单的 LRU：超过 200 条清理
        if len(self._cache) > 200:
            cutoff = time.time() - self.cache_ttl
            self._cache = {k: v for k, v in self._cache.items() if v[0] > cutoff}

        return results

    def rerank(self, query: str, hits: Sequence, top_k: Optional[int] = None,
               text_attr: str = "text") -> list:
        """对 hits 列表重排，按重排分数降序

        - hits: list of dict 或 dataclass，需有 text_attr 字段
        - 返回新列表（不修改入参）
        """
        if not hits:
            return []
        docs = []
        for h in hits:
            if isinstance(h, dict):
                docs.append(h.get(text_attr, "") or "")
            else:
                docs.append(getattr(h, text_attr, "") or "")
        scores = self.score(query, docs)
        scored = []
        for h, s in zip(hits, scores):
            if isinstance(h, dict):
                h2 = dict(h)
                h2["rerank_score"] = float(s)
            else:
                h2 = h
                try:
                    h.rerank_score = float(s)  # type: ignore
                except AttributeError:
                    pass
            scored.append((float(s), h2))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [h for _, h in scored]
        if top_k is not None:
            out = out[:top_k]
        return out

    def __repr__(self):
        return f"Reranker(model={self.model_name}, loaded={self._model is not None}, device={self.device})"