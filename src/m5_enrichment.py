from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, hashlib, json as _json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

ENRICH_MODEL = "gpt-4o-mini"
_client = None

# Enrichment tốn 1 API call/chunk (~9s/chunk khi mạng chậm → 114 chunks ≈ 16 phút).
# Cache theo hash nội dung để re-run pipeline (đổi prompt trả lời, đổi top_k...)
# không phải gọi lại LLM.
_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           ".enrich_cache.json")
_cache: dict | None = None


def _cache_key(text: str, source: str, method: str) -> str:
    return hashlib.sha1(f"{method}|{source}|{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _cache = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            _json.dump(_cache, f, ensure_ascii=False)
    except OSError as e:
        print(f"  ⚠️  Không ghi được enrichment cache: {e}")


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _get_client():
    """OpenAI client dùng chung (None nếu không có API key)."""
    global _client
    if not OPENAI_API_KEY:
        return None
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def _chat(system: str, user: str, max_tokens: int = 200, json_mode: bool = False) -> str | None:
    """Gọi LLM 1 lượt. Trả None nếu không có key hoặc call lỗi (caller tự fallback)."""
    client = _get_client()
    if client is None:
        return None
    try:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(
            model=ENRICH_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI call failed: {e}")
        return None


def _extractive_summary(text: str) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return (". ".join(sentences[:2]) + ".") if sentences else text


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    summary = _chat(
        "Tóm tắt đoạn văn sau trong 1-2 câu ngắn gọn bằng tiếng Việt. "
        "Summary PHẢI ngắn hơn đoạn gốc, chỉ giữ con số và điều kiện quan trọng.",
        text, max_tokens=150,
    )
    # Summary dài hơn bản gốc thì vô nghĩa (chunk quá ngắn) → dùng extractive.
    if summary and len(summary) < len(text):
        return summary
    return _extractive_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    raw = _chat(
        f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
        "Trả về mỗi câu hỏi trên 1 dòng, không đánh số, không giải thích.",
        text, max_tokens=200,
    )
    if raw:
        questions = [q.strip().lstrip("0123456789.-) ") for q in raw.split("\n") if q.strip()]
        if questions:
            return questions[:n_questions]

    # Extractive fallback: biến câu khẳng định thành câu hỏi thô.
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = _chat(
        "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. "
        "Chỉ trả về 1 câu, không mở đầu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", max_tokens=80,
    )
    if context:
        return f"{context}\n\n{text}"

    prefix = f"Trích từ {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    raw = _chat(
        'Trích xuất metadata từ đoạn văn. Trả về JSON: '
        '{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}',
        text, max_tokens=150, json_mode=True,
    )
    if raw:
        try:
            data = _json.loads(raw)
            if isinstance(data, dict):
                return data
        except _json.JSONDecodeError as e:
            print(f"  ⚠️  Metadata JSON parse failed: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    raw = _chat(
        """Phân tích đoạn văn và trả về JSON đúng schema:
{
  "summary": "tóm tắt 1-2 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
        max_tokens=400, json_mode=True,
    )
    if not raw:
        return {}
    try:
        data = _json.loads(raw)
        return data if isinstance(data, dict) else {}
    except _json.JSONDecodeError as e:
        print(f"  ⚠️  Enrichment JSON parse failed: {e}")
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods
    method_tag = "+".join(methods)
    cache = _load_cache()
    n_hits = 0

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")
        key = _cache_key(text, source, method_tag)
        cached = cache.get(key)

        if cached is not None:
            summary = cached.get("summary", "")
            questions = cached.get("questions", [])
            enriched_text = cached.get("enriched_text", text)
            auto_meta = cached.get("auto_metadata", {})
            n_hits += 1
        elif use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        if cached is None:
            cache[key] = {"summary": summary, "questions": questions,
                          "enriched_text": enriched_text, "auto_metadata": auto_meta}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method=method_tag,
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    _save_cache()
    if n_hits:
        print(f"  ♻️  {n_hits}/{len(chunks)} chunks lấy từ cache (không gọi API).", flush=True)
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
