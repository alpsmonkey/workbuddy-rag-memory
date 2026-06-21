"""
chunker guard 测试：整段 RAW_JSON 包裹的 cloud profile 必须不入 RAG
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_text, RAW_JSON_START


def test_whole_paragraph_raw_json_skipped():
    """整段被 RAW_JSON 包裹 → 返回 []（防 cloud profile 污染）"""
    text = """<!-- RAW_JSON_START -->
{
  "uid": "abc-123",
  "memoryBlock": "**工作背景**\\n用户是 SAP FICO 专家...",
  "version": 31
}
<!-- RAW_JSON_END -->"""
    chunks = chunk_text(text)
    assert chunks == [], f"应返回空列表（cloud profile 整体 skip），实际 {len(chunks)} 个"


def test_whole_paragraph_raw_json_with_source_path_skipped():
    """即使传了 source_path，整段 RAW_JSON 仍 skip"""
    raw = "<!-- RAW_JSON_START -->\n{\"x\": 1}\n<!-- RAW_JSON_END -->"
    chunks = chunk_text(raw, source="C:/path/MEMORY.md")
    assert chunks == [], f"应 skip RAW_JSON cloud profile，实际 {len(chunks)}"


def test_raw_json_inline_with_surrounding_text_still_extracted():
    """RAW_JSON 内嵌在普通文本里 → Step 0 抽出，按 key 拆（保留旧行为）"""
    text = """# 项目笔记

<!-- RAW_JSON_START -->
{
  "uid": "abc",
  "summary": "SkillFather 是 Agent Skill 适配度分析工具",
  "key_facts": ["5 维评分", "Python 3.13"]
}
<!-- RAW_JSON_END -->

## 决策
2026-05-28 决定用 Python。"""
    chunks = chunk_text(text)
    # RAW_JSON 拆出 summary / key_facts，加上 H2 "决策" → 至少 2 个
    assert len(chunks) >= 2, f"应保留 inline RAW_JSON 拆段 + 普通 H2，实际 {len(chunks)}"


def test_normal_memory_not_affected():
    """普通 MEMORY.md 内容（含 # 标题）→ 正常切分（不被 guard 误伤）"""
    text = """# 鹏哥偏好

## 编码风格
用 Python 3.13，习惯 5 个/批节奏提问，偏好表格化展示。
"""
    chunks = chunk_text(text)
    assert len(chunks) >= 1, f"普通 MEMORY.md 应正常切分，实际 {len(chunks)} 个"
    assert any("Python 3.13" in c.text for c in chunks), "应保留'Python 3.13'内容"


def test_legitimate_user_memory_md_with_date_not_skipped():
    """含日期但不是 RAW_JSON → 不应被 guard 误跳"""
    text = """## 2026-06-21 进度

决定用保守修法 A。删除不可逆，reclassify 无损。
"""
    chunks = chunk_text(text)
    assert any("保守修法" in c.text for c in chunks), "合法 daily log 应保留"


if __name__ == "__main__":
    test_whole_paragraph_raw_json_skipped()
    test_whole_paragraph_raw_json_with_source_path_skipped()
    test_raw_json_inline_with_surrounding_text_still_extracted()
    test_normal_memory_not_affected()
    test_legitimate_user_memory_md_with_date_not_skipped()
    print("✅ All 5 chunker guard tests passed")
