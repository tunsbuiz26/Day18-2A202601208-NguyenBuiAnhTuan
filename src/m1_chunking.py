from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\n")
HEADER_SPLIT_RE = re.compile(r"(^#{1,3}\s+.+$)", re.MULTILINE)
HEADER_MATCH_RE = re.compile(r"^#{1,3}\s+")

# Model dùng cho semantic chunking — cache ở module level vì load lại rất chậm.
_SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
_semantic_model = None


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _get_semantic_model():
    """Lazy-load embedding model cho semantic chunking (None nếu không load được)."""
    global _semantic_model
    if _semantic_model is None:
        try:
            # HF_HUB_OFFLINE=1: chỉ đọc cache. Model này đã cache sẵn, nhưng nếu để
            # from_pretrained() gọi network thì mạng stall làm treo cả pytest
            # (xem m3_rerank._load_model, m2_search._load_encoder).
            from sentence_transformers import SentenceTransformer
            prev = os.environ.get("HF_HUB_OFFLINE")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                _semantic_model = SentenceTransformer(_SEMANTIC_MODEL_NAME)
            finally:
                if prev is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev
        except Exception as e:  # offline / thiếu model → caller fallback sang chunk_basic
            print(f"  ⚠️  Không load được {_SEMANTIC_MODEL_NAME}: {e}")
            _semantic_model = False
    return _semantic_model or None


def _split_sentences(text: str) -> list[str]:
    """Tách text thành câu: theo dấu câu hoặc dòng trống."""
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s and s.strip()]


def _pack(units: list[str], max_size: int, sep: str = " ") -> list[str]:
    """Gộp các đơn vị text liền kề thành block ≤ max_size ký tự.

    Đơn vị dài hơn max_size bị hard-split để không có block nào vượt ngưỡng.
    """
    blocks: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_size:
            if current:
                blocks.append(current)
                current = ""
            for i in range(0, len(unit), max_size):
                blocks.append(unit[i:i + max_size].strip())
            continue
        candidate = f"{current}{sep}{unit}" if current else unit
        if len(candidate) > max_size and current:
            blocks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        blocks.append(current)
    return [b for b in blocks if b]


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None, min_chunk_size: int = 100) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Cắt chunk khi cosine_sim(câu trước, câu hiện tại) < threshold. `min_chunk_size`
    chặn các chunk vụn (heading 1 dòng, câu ngắn) — chỉ cho phép cắt khi chunk
    hiện tại đã đủ dài, nếu không thì gộp tiếp.
    """
    from numpy import dot
    from numpy.linalg import norm

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0],
                      metadata={**metadata, "chunk_index": 0, "strategy": "semantic"})]

    model = _get_semantic_model()
    if model is None:  # không có embedding model → giữ pipeline chạy được
        return chunk_basic(text, metadata={**metadata, "strategy": "semantic"})

    embeddings = model.encode(sentences)

    def cosine_sim(a, b) -> float:
        return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        current_len = len(" ".join(groups[-1]))
        if sim < threshold and current_len >= min_chunk_size:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(text=" ".join(g).strip(),
              metadata={**metadata, "chunk_index": i, "strategy": "semantic",
                        "n_sentences": len(g)})
        for i, g in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_text in _pack(paragraphs, parent_size, sep="\n\n"):
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "chunk_index": len(parents), "strategy": "hierarchical"},
        ))

        # Child nhỏ hơn → embedding tập trung, retrieve chính xác hơn.
        child_units = _split_sentences(parent_text) or [parent_text]
        for j, child_text in enumerate(_pack(child_units, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child", "parent_id": pid,
                          "child_index": j, "strategy": "hierarchical"},
                parent_id=pid,
            ))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = HEADER_SPLIT_RE.split(text)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    def flush():
        body = f"{current_header}\n\n{current_content}".strip() if current_header else current_content.strip()
        if body:
            chunks.append(Chunk(
                text=body,
                metadata={**metadata, "section": current_header.strip() or "(preamble)",
                          "level": current_header.count("#") if current_header else 0,
                          "chunk_index": len(chunks), "strategy": "structure"},
            ))

    for part in parts:
        if not part or not part.strip():
            continue
        if HEADER_MATCH_RE.match(part):
            flush()                       # đóng section trước khi mở section mới
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part
    flush()

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
