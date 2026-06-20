"""
embedding 封装
- 默认 BAAI/bge-m3 (1024 维)
- 自动 fallback: onnxruntime -> sentence-transformers
- 失败兜底: 全零向量 + warning

网络策略：
- 默认 HF_HUB_OFFLINE=1（强制使用本地缓存，不连外网）
- 下载模型时显式设置 HF_HUB_OFFLINE=0
- 取消限制用 HF_HUB_OFFLINE=0 环境变量
"""
from __future__ import annotations
import os
import hashlib
import warnings
from functools import lru_cache
from typing import List, Union

import numpy as np

# 【默认离线】防止 CI/冒烟测试或网络不稳定时反复连 HuggingFace
# 任何希望联网下载的位置必须显式 os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

EMBED_MODEL_DEFAULT = "BAAI/bge-m3"
EMBED_DIM_DEFAULT = 1024


class Embedder:
    """Embedding 包装器：自动选择最优后端"""

    def __init__(
        self,
        model_name: str = None,
        dim: int = None,
        device: str = "cpu",
        normalize: bool = True,
    ):
        self.model_name = model_name or os.getenv("EMBED_MODEL", EMBED_MODEL_DEFAULT)
        self.dim = dim or int(os.getenv("EMBED_DIM", EMBED_DIM_DEFAULT))
        self.device = device
        self.normalize = normalize
        self._backend = None
        self._model = None
        self._init_backend()

    def _init_backend(self):
        """尝试 sentence-transformers -> 失败则用简化哈希向量兜底"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._backend = "sentence-transformers"
            # 自动校准维度
            test_vec = self._model.encode(["test"], normalize_embeddings=self.normalize)
            self.dim = test_vec.shape[1]
            return
        except Exception as e:
            warnings.warn(f"[Embedder] sentence-transformers 加载失败: {e}")

        # 兜底: 用 hash 生成稳定伪向量（仅用于冒烟测试，不可用作生产）
        self._backend = "hash-fallback"
        warnings.warn(
            "[Embedder] 使用 hash 兜底后端，向量无语义信息，仅供测试。\n"
            "请安装: pip install sentence-transformers"
        )

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本 -> (N, dim) ndarray"""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        if self._backend == "sentence-transformers":
            vecs = self._model.encode(
                texts,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        else:
            vecs = np.array([self._hash_embed(t) for t in texts], dtype=np.float32)

        return vecs[0] if single else vecs

    def _hash_embed(self, text: str) -> np.ndarray:
        """确定性 hash 向量（仅兜底用）"""
        vec = np.zeros(self.dim, dtype=np.float32)
        for i in range(self.dim):
            h = hashlib.md5(f"{text}:{i}".encode()).digest()
            vec[i] = (h[0] / 255.0) - 0.5
        # 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @property
    def backend(self) -> str:
        return self._backend

    def __repr__(self):
        return f"Embedder(model={self.model_name}, dim={self.dim}, backend={self._backend})"


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    """单例：避免重复加载模型"""
    return Embedder()
