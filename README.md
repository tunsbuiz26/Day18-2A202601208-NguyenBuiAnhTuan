# Lab 18: Production RAG Pipeline

**AICB-K34 · Ngày 18 · Production RAG**
**Giảng viên:** Nguyễn Tiến Đồng · **Thời gian:** 2h implement + 30 phút reflection
**Ref**: repo của giảng viên Trần Quang Thiện K2.

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency         | Bắt buộc? | Dùng cho                                       |
| ------------------ | ----------- | ----------------------------------------------- |
| Docker (Qdrant)    | ✅ Có      | M2 Dense Search                                 |
| Python 3.11+       | ✅ Có      | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `OPENAI_API_KEY` | ⚠️ M4+M5  | RAGAS eval (M4), Enrichment LLM (M5)            |

**Pre-download models** (tránh timeout trong lab):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

```bash
git clone <repo-url> && cd lab18-production-rag
docker compose up -d                    # Qdrant
pip install -r requirements.txt
cp .env.example .env                    # Điền API keys
python naive_baseline.py                # ⚠️ Chạy TRƯỚC để có baseline
```

## Chạy toàn bộ

```bash
python main.py                          # Naive + Production + So sánh
python check_lab.py                     # Kiểm tra trước khi nộp
```

## Cấu trúc repo

```
lab18-production-rag/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 40 .md files + PDFs
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (40 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_13-2023.pdf   # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 30 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép nhóm
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (nhóm)
│   ├── group_report.md         # Báo cáo nhóm
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (sau khi chạy main.py)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    ├── failure_analysis.md
    └── group_report.md
```

## Timeline

| Thời gian | Hoạt động                               |
| ---------- | ------------------------------------------ |
| 0:00–0:10 | Setup + chạy`naive_baseline.py`         |
| 0:10–1:40 | Implement M1 → M2 → M3 → M4 → M5       |
| 1:40–2:00 | Chạy pipeline + RAGAS + failure analysis  |
| 2:00–2:30 | Reflection: lecture mapping + project plan |
