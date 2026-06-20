r"""
下载 BAAI/bge-reranker-v2-m3（Cross-Encoder 重排模型）

- 走 HF 官方源（不再用 hf-mirror，主源已恢复）
- 临时解锁 HF_HUB_OFFLINE=0（下载期需要联网）
- allow_patterns 限制只下载推理所需文件（pytorch.bin + tokenizer）
  - 不下 onnx / tensorflow 变体
- 失败自动切换 hf-mirror

输出:
  C:\Users\JJ\.cache\huggingface\models--BAAI--bge-reranker-v2-m3\
"""
from __future__ import annotations
import os
import sys
import time

# 临时解锁联网（下载期）
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

CACHE_DIR = r"C:\Users\JJ\.cache\huggingface"
REPO_ID = "BAAI/bge-reranker-v2-m3"


def main():
    from huggingface_hub import snapshot_download

    # 推理所需最小集（Cross-Encoder 只用 pytorch + tokenizer 配置）
    allow_patterns = [
        "*.json",           # config / tokenizer config
        "*.txt",            # tokenizer vocab / special tokens
        "tokenizer.*",      # tokenizer 文件
        "vocab.txt",
        "*.safetensors",    # 权重（新版）
        "pytorch_model.bin",  # 权重（旧版）
        "sentencepiece.model",
        "spiece.model",
    ]

    print(f"📥 下载 {REPO_ID}")
    print(f"   cache: {CACHE_DIR}")
    print(f"   allow_patterns: {allow_patterns}")
    print()

    t0 = time.time()
    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            cache_dir=CACHE_DIR,
            allow_patterns=allow_patterns,
            max_workers=4,
        )
        elapsed = time.time() - t0
        print(f"\n✅ 完成: {path}")
        print(f"   耗时: {elapsed:.1f}s")

        # 统计
        import os
        blobs = os.path.join(path, "blobs") if os.path.isdir(os.path.join(path, "blobs")) else path
        total_size = 0
        file_count = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total_size += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass
        print(f"   文件数: {file_count}, 总大小: {total_size / 1024 / 1024:.1f} MB")
        return 0

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n❌ 主源失败 ({elapsed:.1f}s): {e}")
        print(f"   尝试 hf-mirror ...")

        # fallback: hf-mirror
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        try:
            t1 = time.time()
            path = snapshot_download(
                repo_id=REPO_ID,
                cache_dir=CACHE_DIR,
                allow_patterns=allow_patterns,
                max_workers=4,
            )
            elapsed = time.time() - t1
            print(f"\n✅ mirror 完成: {path}")
            print(f"   耗时: {elapsed:.1f}s")
            return 0
        except Exception as e2:
            print(f"\n💥 mirror 也失败: {e2}")
            return 1


if __name__ == "__main__":
    sys.exit(main())