"""
chunker 测试
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_text, extract_entities, extract_project, compute_confidence


def test_basic_chunk():
    text = """# 项目笔记

## SkillFather 架构
SkillFather 用 Python 开发，核心是 5 维评分。

## 决策记录
2026-05-28 弃用 GraphQL，理由是复杂度高于收益。
"""
    chunks = chunk_text(text)
    assert len(chunks) >= 2, f"应切出至少 2 个 chunk，实际 {len(chunks)}"
    # 第一段应包含"SkillFather"
    assert any("SkillFather" in c.text for c in chunks)


def test_skip_short():
    text = "短"
    chunks = chunk_text(text)
    assert len(chunks) == 0


def test_long_paragraph_split():
    text = "这是第一句。" * 100
    chunks = chunk_text(text, max_length=200)
    assert all(len(c.text) <= 600 for c in chunks)  # 允许 1.x 倍


def test_extract_entities():
    entities = extract_entities("我用了 Python 和 lancedb 搭 RAG 系统")
    assert "python" in entities
    assert "lancedb" in entities


def test_extract_project():
    proj = extract_project("我们决定在 SkillFather 项目里用 Python")
    assert proj is not None


def test_confidence_high_for_decision():
    c1 = compute_confidence("2026-05-28 决定弃用 GraphQL，复杂度太高")
    c2 = compute_confidence("我今天喝了一杯咖啡")
    assert c1 > c2


def test_metadata_extraction():
    chunks = chunk_text("SkillFather 项目决定用 Python 做后端开发，2026-05-28 落地")
    assert len(chunks) >= 1
    c = chunks[0]
    assert "python" in c.meta["entities"]
    assert c.meta["confidence"] > 0.5
    assert c.meta["ts"]  # 有时间戳


if __name__ == "__main__":
    test_basic_chunk()
    test_skip_short()
    test_long_paragraph_split()
    test_extract_entities()
    test_extract_project()
    test_confidence_high_for_decision()
    test_metadata_extraction()
    print("✅ chunker 全部 7 个测试通过")
