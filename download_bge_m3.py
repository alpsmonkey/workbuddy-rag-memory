"""
预下载 bge-m3 模型到指定缓存目录
- 默认 HF 缓存路径: C:\\Users\\JJ\\.cache\\huggingface
- 可通过 HF_HOME 环境变量自定义
- 自动 fallback: 官方源失败 → hf-mirror 镜像
"""
import os
import sys
from pathlib import Path

# 【下载阶段必须离线=0】embedder.py 默认把 HF_HUB_OFFLINE 锁为 1
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

CACHE_DIR = Path(os.environ["HF_HOME"])
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 模型权重（只下 pytorch 版，跳过 onnx/ColBERT/sparse linear，节省 50% 体积）
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "sentencepiece*",
    "*.model",
    "1_Pooling/*",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "model.safetensors",
]


def try_download(endpoint: str = None, label: str = "official") -> bool:
    """尝试一次下载；返回是否成功"""
    print(f"\n[start] 通过 {label} 源下载 BAAI/bge-m3 到: {CACHE_DIR}")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id="BAAI/bge-m3",
            cache_dir=str(CACHE_DIR),
            allow_patterns=ALLOW_PATTERNS,
            max_workers=4,
        )
        print(f"[done] 模型已下载到: {path}")
        files = list(Path(path).rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"[verify] {len(files)} 个文件，总大小: {total_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"[FAIL] {label} 源失败: {e}")
        return False


# 主流程：先官方，后镜像
success = try_download(endpoint=None, label="官方 (huggingface.co)")
if not success:
    print("[fallback] 切换到 hf-mirror 镜像...")
    success = try_download(endpoint="https://hf-mirror.com", label="hf-mirror")

if not success:
    print("\n[FAIL] 所有源都失败")
    print("  可手动指定镜像: set HF_ENDPOINT=https://hf-mirror.com && python download_bge_m3.py")
    sys.exit(1)
