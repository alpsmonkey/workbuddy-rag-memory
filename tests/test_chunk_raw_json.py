"""
test_chunk_raw_json.py
- 验证 RAW_JSON 块被按 key 拆分成多个 chunk
- 验证嵌入主流程不会破坏 JSON 拆分结果
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_text


def test_raw_json_block_split():
    """RAW_JSON 块应被按 key 拆分（而不是整块一个 chunk）"""
    text = """# 测试文档

普通段落：这是一个普通段落，应该单独成一个 chunk。

<!-- RAW_JSON_START -->
{
  "memory_type": "user_preference",
  "summary": "鹏哥偏好简洁直接、表格化展示、Python 3.13",
  "key_facts": ["用 Python 3.13", "5个/批节奏提问"],
  "tags": ["python", "sap", "fico"]
}
<!-- RAW_JSON_END -->

JSON 块后的普通段落，应该单独成一个 chunk。
"""

    chunks = chunk_text(text, source="test")
    print(f"Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  [{i}] block_type={c.meta.get('block_type', 'text')}  text={c.text[:60]}")

    # 至少应该有 4 个：普通段落 1 + JSON 块拆出 4 个 + JSON 后普通段落 1
    assert len(chunks) >= 5, f"应至少 5 个 chunk，实际 {len(chunks)}"

    # 找到 raw_json 类型的 chunks
    json_chunks = [c for c in chunks if c.meta.get("block_type") == "raw_json"]
    assert len(json_chunks) >= 3, f"JSON 块应至少拆出 3 个 chunk，实际 {len(json_chunks)}"

    # 应该包含 summary / key_facts / tags 拆分
    text_concat = " ".join(c.text for c in json_chunks)
    assert "summary:" in text_concat
    assert "key_facts:" in text_concat
    assert "tags:" in text_concat
    assert "memory_type:" in text_concat

    print("✅ test_raw_json_block_split passed")


def test_raw_json_invalid_falls_back():
    """无效 JSON 应退回为单个 chunk"""
    text = """<!-- RAW_JSON_START -->
这不是合法 JSON {{{ broken
<!-- RAW_JSON_END -->"""
    chunks = chunk_text(text)
    # 至少 1 个 chunk（退回处理）
    assert len(chunks) >= 1
    print(f"✅ test_raw_json_invalid_falls_back passed ({len(chunks)} chunk)")


def test_normal_text_no_regression():
    """普通文本切分不受 RAW_JSON 改造影响"""
    text = """# 标题

## 二级标题
第一段内容。第二段内容。

## 另一个
第三段。
"""
    chunks = chunk_text(text)
    # 不应该有 raw_json 块
    json_chunks = [c for c in chunks if c.meta.get("block_type") == "raw_json"]
    assert len(json_chunks) == 0, "普通文本不应出现 raw_json 类型"
    print(f"✅ test_normal_text_no_regression passed ({len(chunks)} chunks)")


if __name__ == "__main__":
    test_raw_json_block_split()
    test_raw_json_invalid_falls_back()
    test_normal_text_no_regression()
    print("\n✅ all chunker RAW_JSON tests passed")