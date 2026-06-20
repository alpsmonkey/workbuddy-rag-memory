"""
BGE Reranker v2-m3 包装
- 直接用 transformers AutoModel（避免 CrossEncoder 的 HF_HUB_OFFLINE 兼容问题）
- Lazy load + 兼容 HF_HUB_OFFLINE=1
- 返回 sigmoid 归一化分数 [0, 1]
"""
from __future__ import annotations
import os
from typing import List, Optional, Sequence

# 与 embedder.py 保持一致：默认离线
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


class Reranker:
    """BGE Reranker v2-m3 cross-encoder"""

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str = None, max_length: int = 512):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cuda":
            self._model = self._model.to(self._device)

    @staticmethod
    def _sigmoid(x) -> float:
        try:
            import math
            return 1.0 / (1.0 + math.exp(-float(x)))
        except Exception:
            return 0.5

    def score(self, query: str, documents: List[str]) -> List[float]:
        """对 (query, doc) 对打分 → sigmoid 后 [0, 1]
        Returns 与 documents 等长的归一化分数列表

        顺序循环（cross-encoder 实测更稳）：
        - micro-bench: n=1=197ms, n=5=82ms/doc, n=10=67ms/doc, n=20=66ms/doc
        - 但真实 query 文档长度差异大，padding 把 batching 收益吃光
        - 实测：批量 20 docs = 10s；顺序 20 docs = 6s
        """
        if not documents:
            return []
        self._load()
        import torch
        scores: List[float] = []
        with torch.no_grad():
            for doc in documents:
                inputs = self._tokenizer(
                    [[query, doc]],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                if self._device == "cuda":
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                logits = self._model(**inputs).logits.squeeze().item()
                scores.append(self._sigmoid(logits))
        return scores

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
                except Exception:
                    pass
            scored.append((float(s), h2))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [h for _, h in scored]
        if top_k is not None:
            out = out[:top_k]
        return out

    def __repr__(self):
        return f"Reranker(model={self.model_name}, loaded={self._model is not None})"