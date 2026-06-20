"""
bge-m3 模型下载脚本（纯 urllib，绕开 hf_hub 库的不稳定 HEAD 请求）
- hf-mirror 镜像
- 流式下载 + 进度条
- 失败重试
"""
import os
import sys
import time
import urllib.request
import urllib.error
import socket
from pathlib import Path

# 【下载阶段】显式关 offline（embedder 默认是 1）
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
socket.setdefaulttimeout(60)

CACHE_DIR = Path(r"C:\Users\JJ\.cache\huggingface\hub\bge-m3--custom")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BLOBS = CACHE_DIR / "blobs"
BLOBS.mkdir(exist_ok=True)
SNAPSHOTS = CACHE_DIR / "snapshots"
SNAPSHOTS.mkdir(exist_ok=True)

BASE_URL = "https://hf-mirror.com/BAAI/bge-m3/resolve/main"

# 必需文件（按重要性排序）
FILES = [
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
    "model.safetensors",  # 主权重
]


def fmt_size(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "=" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {fmt_size(downloaded)}/{fmt_size(total_size)}")
    sys.stdout.flush()


def download_file(filename, max_retry=3):
    blob_path = BLOBS / f"{filename}.blob"
    if blob_path.exists() and blob_path.stat().st_size > 100:
        print(f"  [skip] {filename} 已存在 ({fmt_size(blob_path.stat().st_size)})")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"  [get]  {filename}")
    for attempt in range(1, max_retry + 1):
        try:
            urllib.request.urlretrieve(url, blob_path, reporthook=progress_hook)
            print()  # 换行
            size = blob_path.stat().st_size
            print(f"  [ok]   {filename}: {fmt_size(size)}")
            return True
        except (urllib.error.URLError, socket.timeout) as e:
            print(f"\n  [retry] {filename} attempt {attempt}/{max_retry}: {e}")
            if attempt < max_retry:
                time.sleep(2)
    return False


def link_to_snapshot(filename):
    """在 snapshots 目录建符号链接（HF 库要求的目录结构）"""
    blob = BLOBS / f"{filename}.blob"
    snap = SNAPSHOTS / filename
    if snap.exists() or snap.is_symlink():
        try:
            snap.unlink()
        except Exception:
            pass
    try:
        snap.symlink_to(blob)
    except OSError:
        # Windows 权限问题用复制
        import shutil
        shutil.copy2(blob, snap)
    return snap


def main():
    print("=" * 60)
    print(f"目标模型: BAAI/bge-m3 (2.3GB)")
    print(f"缓存目录: {CACHE_DIR}")
    print(f"镜像源:   {BASE_URL}")
    print("=" * 60)
    print()

    failed = []
    for f in FILES:
        if not download_file(f):
            failed.append(f)
        print()

    print("=" * 60)
    if failed:
        print(f"[FAIL] 失败 {len(failed)} 个文件: {failed}")
        return 1
    print(f"[OK] 全部 {len(FILES)} 个文件下载完成")

    # 建 snapshot 链接
    for f in FILES:
        link_to_snapshot(f)

    # 验证
    total = sum((BLOBS / f"{f}.blob").stat().st_size for f in FILES if (BLOBS / f"{f}.blob").exists())
    print(f"[verify] 总大小: {fmt_size(total)}")
    print(f"[path]   快照路径: {SNAPSHOTS}")
    print()
    print("如需让 sentence-transformers 用本地路径，可设置:")
    print(f'  set SENTENCE_TRANSFORMERS_HOME={CACHE_DIR}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
