# Group Report — Lab 18: Production RAG

**Nhóm:** làm cá nhân (1 người)
**Ngày:** 18/08/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Nguyễn Bùi Anh Tuấn — 2A202601208 | M1: Chunking | ☑ | 13/13 |
| Nguyễn Bùi Anh Tuấn — 2A202601208 | M2: Hybrid Search | ☑ | 5/5 |
| Nguyễn Bùi Anh Tuấn — 2A202601208 | M3: Reranking | ☑ | 5/5 |
| Nguyễn Bùi Anh Tuấn — 2A202601208 | M4: Evaluation | ☑ | 4/4 |
| Nguyễn Bùi Anh Tuấn — 2A202601208 | M5: Enrichment + `pipeline.py` | ☑ | 10/10 |

`pytest tests/ -q` → **37 passed (~65s)** · `# TODO:` còn lại trong `src/`: **0**.

> Ghi chú khi chạy `check_lab.py`: mục "Auto-tests" báo `pytest error: invalid literal for int()`
> — đây là lỗi **parse** của chính `check_lab.py` (nó `int()` token đầu của dòng summary cuối),
> trong khi pytest 9.x in dòng đó có viền `=====` (`====== 37 passed in 64.55s ======`).
> Chạy trực tiếp `pytest tests/ -v` vẫn 37/37 pass. Tôi giữ nguyên `check_lab.py` như đề bài phát.

**Cấu hình production** (`config.py`): chunk hierarchical parent 2048 / child 256 ·
`EXPAND_TO_PARENT=True` · BM25 + Dense mỗi bên top-20 → RRF `k=60` → rerank top-3 ·
embedding `paraphrase-multilingual-MiniLM-L12-v2` (override qua `.env`) · Qdrant embedded local ·
LLM + RAGAS judge đều `gpt-4o-mini`, embeddings cho RAGAS chạy local.

## Kết quả RAGAS

20 câu (`test_set.json`). Naive = `chunk_basic(500)` + dense-only + prompt 1 dòng.
Production = hierarchical → enrichment (1 API call/chunk) → hybrid+RRF → rerank → prompt v3.

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8000 | 0.7792 | −0.0208 |
| Answer Relevancy | 0.6283 | 0.7110 | +0.0827 |
| Context Precision | 0.9417 | 0.9750 | +0.0333 |
| Context Recall | 0.8250 | 0.9250 | +0.1000 |

Latency/query: retrieval 18.1ms · rerank 1.4ms · generation 2120ms (p95 2971ms) → **99.1% ở LLM**.
Build 1 lần: chunking 0.12s · enrichment 411s (lần đầu; sau đó 0.01s nhờ cache) · indexing 28.3s.

## Key Findings

1. **Biggest improvement — context_recall +0.10 nhờ hierarchical small-to-big.** Match bằng child
   256 ký tự (embedding tập trung, ít nhiễu) nhưng trả parent 2048 cho LLM. So với baseline riêng
   phần chunking, đây là thay đổi 1 dòng config nhưng ăn điểm nhiều nhất — và nó cũng làm
   answer_relevancy tăng theo vì answer không còn bị cắt giữa ý.
2. **Biggest challenge — hạ tầng, không phải thuật toán.** RAGAS mặc định gọi
   `text-embedding-ada-002` → 403 (key chỉ có `gpt-4o-mini`) và trả **NaN thay vì raise**, nên
   report hiện faithfulness 0.0 như thể model dở. Phải tự inject judge
   (`gpt-4o-mini` + `HuggingFaceEmbeddings` local, `RunConfig(timeout=900)`) mới có số thật:
   smoke test đi từ `{F 0.0, AR 0.0}` / timeout 180s → `{F 1.0, AR 0.97}` / 4s. Cùng loại lỗi:
   cross-encoder treo pipeline >420s vì HF Hub stall → phải check cache trước khi
   `from_pretrained()` (`_weights_cached()` + `HF_HUB_OFFLINE=1`) → test_m3 về 0.28s.
3. **Surprise finding — siết prompt làm điểm TỆ ĐI, và faithfulness đánh đổi với answer_relevancy.**
   A/B 4 phiên bản prompt trên cùng index (`reports/ab_prompt_runs/`): prompt template 1 dòng cho
   F 0.8125 / AR 0.6387; prompt siết "cấm suy luận + bắt buộc từ chối" tụt xuống 0.7333 / 0.6871
   (over-refusal 2 câu → F = 0); prompt v3 cân bằng đạt 0.7792 / **0.7110**; ép trả lời 1 câu
   (v4) đẩy AR lên 0.7148 nhưng F rơi về 0.7125. Càng ngắn → AR↑ F↓, vì answer mất phần dẫn số
   gốc từ context. Ngoài ra CP/CR gần như bất biến giữa các prompt (đúng định nghĩa), và chênh
   0.025 giữa các lần chạy = **nhiễu của LLM-as-judge** → không tin cải thiện < 0.03.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** 3/4 metric tăng, mạnh nhất là context_recall
   +0.10 và answer_relevancy +0.083. Faithfulness giảm 0.02 — giải thích: production trả lời trọn
   vẹn nên **nhiều claim hơn**, mà faithfulness phạt theo số claim; naive trả lời cụt nên "dễ"
   faithful. Phải đọc 4 metric cùng nhau.
2. **Biggest win — M1 (hierarchical) + M2 (hybrid RRF).** M1 đưa context_recall lên 0.925;
   M2 giải bài toán tiếng Việt: `underthesea` trả `nghỉ_phép` còn `BM25Okapi` tokenize bằng
   `.split()` → phải `.replace("_", " ")` cho **cả corpus và query**, không thì BM25 recall = 0
   với mọi từ ghép. RRF chỉ dùng thứ hạng nên không cần normalize BM25 (không chặn trên) với
   cosine ([-1,1]).
3. **Case study — "Thông tin lương thuộc cấp độ phân loại dữ liệu nào?"** (chi tiết trong
   `analysis/failure_analysis.md`): câu multi-hop cần `ky_luong.md` + `phan_loai_du_lieu.md`.
   Top-4 retrieval đều là child của **cùng 1 parent** → `_expand_to_parent()` dedup còn **1
   context, 1 nguồn** → LLM từ chối → F 0 + AR 0, dù CP/CR đều 1.0. Bài học: small-to-big giúp
   recall nhưng **giết diversity**; và CP/CR cao không đảm bảo đủ context cho multi-hop.
4. **Next optimization nếu có thêm 1 giờ:** (a) bù child chunk khác `source` khi dedup parent làm
   context < 3; (b) thêm `effective_date`/`version` vào auto-metadata + filter Qdrant để chặn
   chunk phiên bản cũ (đang làm mất CP ở 2 câu); (c) tách câu tính toán ra chấm bằng
   answer_correctness thay vì faithfulness; (d) tải được cross-encoder thật — cả 5 run hiện đều
   chạy nhánh fallback nên điểm đang là **retrieval-only**.
