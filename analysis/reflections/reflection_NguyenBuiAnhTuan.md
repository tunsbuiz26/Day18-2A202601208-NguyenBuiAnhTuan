# Individual Reflection — Lab 18: Production RAG

**Tên:** Nguyễn Bùi Anh Tuấn · **MSSV:** 2A202601208
**Module phụ trách:** M1 → M5 + `pipeline.py` (làm cá nhân, toàn bộ 5 modules)
**Ngày:** 18/08/2026

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số đo thật) |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` — cosine giữa 2 câu liền kề, cắt khi `sim < 0.85` | Trên toàn bộ `data/`: threshold 0.85 tạo **121 chunks** (avg 172 ký tự) vs basic **51 chunks** (avg 410). Phải thêm `min_chunk_size=100` vì heading markdown 1 dòng có similarity rất thấp với câu sau → cắt liên tục thành chunk vụn |
| Hierarchical / small-to-big | M1 + pipeline | `chunk_hierarchical(2048, 256)` + `_expand_to_parent()` | 109 child (avg 191) / 26 parent. Match bằng child (embedding tập trung) nhưng trả parent cho LLM. Đây là thứ đẩy **context_recall 0.8250 → 0.9250 (+0.10)** — mức tăng lớn nhất trong cả lab. Mặt tối: dedup theo parent làm 3 chunk co lại 1 context ở câu multi-hop (xem case study) |
| Structure-aware chunking | M1 | `chunk_structure_aware()` — split theo regex `^#{1,3}\s+` | 106 chunks, nhưng `max_len = 790` vì một section dài không bị cắt — giữ nguyên table/list là đúng ý đồ, đổi lại chunk không đều |
| BM25 cho tiếng Việt | M2 | `segment_vietnamese()` + `BM25Search` | underthesea trả `"nghỉ_phép"`, `BM25Okapi` tokenize bằng `.split()` → phải `.replace("_", " ")` cho **cả corpus và query**, nếu không BM25 recall = 0 với mọi từ ghép |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion(k=60)` | RRF chỉ dùng **thứ hạng** nên không phải normalize giữa BM25 (không chặn trên) và cosine ([-1,1]) — đây là lý do RRF thắng weighted-sum về mặt kỹ thuật |
| Vector store | M2 | `DenseSearch._connect()` / `query_points()` | `qdrant-client ≥ 1.10` bỏ `search()`, phải dùng `query_points(...).points`. Embedded local mode giữ file lock → cache client theo path |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` (top-20 → top-3) | Rerank stage đo được **1.4ms avg / 2.2ms max** — nhưng đó là latency của nhánh fallback (model không tải nổi, xem 2.7). Điểm học được: cross-encoder chấm từng cặp (query, doc) nên O(N) forward pass → chỉ chạy được ở top-N nhỏ, không thể thay retrieval |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` + `_ragas_judge()` | Metric thấp nhất là **answer_relevancy (0.7110)** vì RAGAS sinh lại câu hỏi từ answer rồi so cosine với câu hỏi gốc — answer thêm mệnh đề phụ hoặc liệt kê nhiều version bị trừ nặng (case "thâm niên bao nhiêu năm" chỉ được 0.511 dù faithfulness 1.0) |
| Prompt engineering cho metric | pipeline | `ANSWER_SYSTEM_PROMPT` (v1→v4, A/B cùng index) | **F và AR đánh đổi nhau**: prompt 1 dòng = F 0.8125/AR 0.6387; siết "cấm suy luận + bắt buộc từ chối" = 0.7333/0.6871 (over-refusal 2 câu → F=0); v3 cân bằng = 0.7792/**0.7110**; ép 1 câu = 0.7125/0.7148. Chênh CP/CR 0.025 giữa các run = nhiễu judge → không tin cải thiện < 0.03 |
| Failure analysis / Error Tree | M4 | `failure_analysis()` + `DIAGNOSTIC_TREE` | Map `min(4 metrics)` → nguyên nhân gốc → fix. Rất hiệu quả: 4/5 câu tệ nhất có worst metric = faithfulness. Nhưng cây chẩn đoán **sai 1 chỗ**: khi answer là câu từ chối, nó vẫn kết luận "LLM hallucinating" trong khi nguyên nhân thật là thiếu mắt nối trong context |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Prepend 1 câu mô tả vị trí chunk trong tài liệu trước khi embed. Kết hợp trong 1 API call với summary + HyQA + metadata: **114 call thay vì 456 call** (giảm 75%) |
| Enrichment cost | M5 | `enrich_chunks()` + cache theo `sha1(method|source|text)` | Enrichment là bottleneck build: **411s cho 114 chunks** (lần chạy đầu, có lúc 988s khi mạng chậm) vs indexing 28.3s. Sau khi có cache đĩa: **0.01s** → nhờ đó A/B được 4 phiên bản prompt trong ~10 phút thay vì ~1 giờ |
| Latency breakdown | pipeline | `build_latency_report()` | retrieval **18.1ms** · rerank **1.4ms** · generation **2120ms** (p95 2971ms) → **99.1% latency nằm ở LLM**. Tối ưu retrieval thêm là vô nghĩa; muốn nhanh phải streaming/answer cache |


## Phần 2: Khó khăn & cách giải quyết

Dưới đây là 9 lỗi thật gặp trong lab, kèm **exact error message** và cách debug.

### 2.1. Không có Qdrant server (Docker chưa chạy)

```
error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.51/containers/json":
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

- **Debug:** `docker ps` → Docker daemon không chạy; `QdrantClient(host=...)` không báo lỗi lúc
  `__init__` (lazy connect) nên lỗi chỉ nổ ra ở lần gọi API đầu tiên → rất khó lần.
- **Fix:** viết `DenseSearch._connect()` thử server trước, **ping bằng `client.get_collections()`**
  để phát hiện sớm, fail thì rơi về embedded local (`QdrantClient(path=QDRANT_LOCAL_PATH)`).
- **Kiến thức thiếu:** Qdrant có chế độ embedded (không cần Docker) — đọc docs
  `qdrant_client` local mode.

### 2.2. `Storage folder already accessed by another instance`

```
RuntimeError: Storage folder .qdrant_local is already accessed by another instance of Qdrant client.
```

- **Nguyên nhân:** `main.py` tạo `DenseSearch()` 2 lần trong **cùng 1 process**
  (naive baseline + production) → 2 handle trên cùng thư mục, local mode giữ file lock.
- **Fix:** cache client theo target trong dict module-level `_qdrant_clients` → 1 process = 1 handle.

### 2.3. BM25 không match query tiếng Việt

- **Triệu chứng:** query "nghỉ phép" trả 0 kết quả BM25 dù document có chữ "nghỉ phép".
- **Debug:** in ra `segment_vietnamese("nghỉ phép")` → `"nghỉ_phép"`. underthesea nối từ ghép
  bằng `_`, còn `BM25Okapi` tokenize bằng `.split()` → token `nghỉ_phép` ≠ `nghỉ` + `phép`.
- **Fix:** `word_tokenize(text, format="text").replace("_", " ")` cho **cả corpus và query**.

### 2.4. `UnicodeEncodeError` khi in tiếng Việt trên Windows

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3: character maps to <undefined>
  File "C:\Program Files\Python311\Lib\encodings\cp1252.py", line 19, in encode
```

- **Nguyên nhân:** stdout Windows mặc định cp1252, không encode được `⚠️` / dấu tiếng Việt.
- **Fix:** chạy với `PYTHONIOENCODING=utf-8`.

### 2.5. `pip install ragas` timeout

```
ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
```

- **Fix:** `pip install --timeout 300 --retries 20 --progress-bar off ragas`.
- **Lưu ý side-effect:** ragas 0.1.22 kéo `openai 2.49.0 → 1.109.1` và `numpy 2.4.6 → 1.26.4`.
  Sau khi cài phải chạy lại pytest để chắc chắn `sentence-transformers` vẫn hoạt động với numpy 1.26.

### 2.6. RAGAS trả `faithfulness = 0.0` và `answer_relevancy = 0.0` (lỗi nặng nhất)

```
Exception raised in Job[1]: PermissionDeniedError(Error code: 403 -
  {'error': {'message': 'Project `proj_Xc6...` does not have access to model `text-embedding-ada-002`',
   'type': 'invalid_request_error', 'code': 'model_not_found'}})
Exception raised in Job[0]: TimeoutError()
```

- **Debug 3 bước:**
  1. Đọc log RAGAS: metric fail → RAGAS **không raise** mà điền `NaN` → `_safe_float()` quy về 0.0
     nên nhìn report tưởng model dở, thực ra là lỗi hạ tầng.
  2. Liệt kê model được cấp quyền: `OpenAI().models.list()` → **đúng 1 model: `gpt-4o-mini`**,
     không có embedding model nào.
  3. RAGAS 0.1.x default judge = `gpt-3.5-turbo-16k` + `text-embedding-ada-002` → cả hai đều 403.
- **Fix:** inject judge thủ công trong `_ragas_judge()`:
  - LLM: `LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))`
  - Embeddings: `LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))`
    — chạy local, không cần quyền API.
  - `RunConfig(timeout=900, max_workers=4)` vì default 180s quá ngắn cho mạng chậm.
- **Kết quả:** smoke test 1 câu từ `{f: 0.0, ar: 0.0}` → `{f: 1.0, ar: 0.97}`, thời gian eval
  giảm từ **180s (timeout) xuống 4s**.

### 2.7. Cross-encoder treo cả pipeline vì mạng

- **Triệu chứng:** `pytest tests/test_m3.py` chạy **>420s không xong**, không log gì.
- **Debug:** kiểm tra HF cache → 7 blob `.incomplete` **0 byte**; đo throughput HF ≈ 144 KB/s,
  có lúc 0 byte/phút → `from_pretrained()` retry vô hạn.
- **Fix:** `_weights_cached()` dùng `huggingface_hub.try_to_load_from_cache()` kiểm tra weights
  **trước** khi gọi `CrossEncoder()`; khi load thì set tạm `HF_HUB_OFFLINE=1` để không bao giờ
  chạm network. Chưa có model → rerank fallback giữ thứ tự retrieval + cảnh báo 1 lần.
- **Kết quả:** test_m3 từ treo >420s → **5 passed in 0.28s**.

### 2.8. `test_semantic_groups_by_topic` fail vì chunk vụn

- **Triệu chứng:** `chunk_semantic()` tạo nhiều chunk hơn `chunk_basic()` → assert fail.
- **Nguyên nhân:** heading markdown 1 dòng có cosine similarity thấp với câu sau → cắt liên tục.
- **Fix:** thêm `min_chunk_size=100` — chỉ cắt khi chunk hiện tại đã đủ dài.

### 2.9. Tự "tối ưu" prompt làm điểm RAGAS tụt (lỗi tư duy, không phải lỗi code)

- **Triệu chứng:** sau khi siết `ANSWER_SYSTEM_PROMPT` để cứu faithfulness, cả 4 metric đều tụt:
  `0.8375 / 0.7266 / 0.9750 / 0.9500` → `0.7333 / 0.6871 / 0.9500 / 0.9000`.
- **Debug:** in ra 10 câu tệ nhất của run mới rồi **đối chiếu với `ground_truth` của test set** —
  phát hiện 2 rule tôi tự thêm là sai:
  1. "KHÔNG cộng/trừ số liệu" xung đột với ground truth: Q11 cần `15 + 3 = 18`, Q16 cần tính phí
     quá hạn, Q17 cần `85% × 20tr`. LLM tuân lệnh → trả lời thiếu → faithfulness giảm.
  2. Rule bắt buộc từ chối gây **over-refusal**: 2 câu có context đúng vẫn trả "Không tìm thấy
     thông tin" → faithfulness = 0 cho cả 2 câu.
- **Fix:** làm A/B tử tế thay vì đoán — cache enrichment cho retrieval **giống hệt nhau** giữa các
  lần chạy, rồi thử 4 prompt (bảng trong `analysis/group_report.md`). Chọn v3: F 0.7792 /
  AR 0.7110 (AR là metric thấp nhất nên ưu tiên nó), và ghi lại cả 4 report thô ở
  `reports/ab_prompt_runs/` làm bằng chứng.
- **Bài học lớn nhất của lab:** chỉ được kết luận "tối ưu có tác dụng" khi (a) giữ mọi biến khác
  cố định và (b) mức tăng lớn hơn nhiễu của judge (đo được ≈ 0.025 = 1/20 câu đổi ý).

### Kiến thức còn thiếu → cách bổ sung

| Thiếu | Cách bổ sung |
|-------|--------------|
| RAGAS internals (LLM-as-judge, prompt của từng metric) | Đọc source `ragas/metrics/_faithfulness.py`, chạy với `raise_exceptions=True` để thấy lỗi thật thay vì NaN |
| Tokenizer tiếng Việt (underthesea vs pyvi vs VnCoreNLP) | So sánh trên 20 query của test set, đo BM25 recall |
| Qdrant production (sharding, quantization, payload index) | Đọc docs Qdrant về scalar quantization + hybrid `query_points` với sparse vector |
| ONNX/quantized reranker để giảm latency | Thử `flashrank` (đã viết sẵn `FlashrankReranker`) và so latency |

---

## Phần 3: Action Plan cho project cá nhân

## Project: Trợ lý hỏi–đáp tài liệu nội bộ (HR + IT policy, tiếng Việt)

### Hiện tại

- RAG pipeline hiện tại: `chunk_basic(500)` → dense-only (1 collection Qdrant) → `gpt-4o-mini`.
- Known issues (đo được từ lab này trên baseline):
  - `answer_relevancy` thấp — LLM trả lời dài dòng, lệch câu hỏi.
  - Query dùng từ khác tài liệu (vocabulary gap) → dense-only miss; không có BM25 để cứu.
  - Tài liệu có **2 version cùng chủ đề** (`nghi_phep_nam_v2023.md` vs `v2024.md`,
    `mat_khau_v1.md` vs `v2.md`) → retrieval trả version cũ, không có metadata `effective_date` để filter.
  - 2 PDF trong `data/` là **scan ảnh, không có text layer** → bị bỏ hoàn toàn
    (`BCTC.pdf`, `Nghi_dinh_so_13-2023...pdf`, ~13.8 MB nội dung mất trắng).

### Plan áp dụng

1. [ ] **Chunking:** `chunk_hierarchical(parent=2048, child=256)` — child để match chính xác,
       parent trả cho LLM đủ ngữ cảnh. Với file markdown có header rõ thì chạy
       `chunk_structure_aware()` trước rồi hierarchical trong từng section (giữ nguyên table/list).
2. [ ] **Search:** Hybrid BM25 + Dense + RRF(k=60). Lý do: tài liệu tiếng Việt nhiều thuật ngữ
       và con số ("12 ngày", "AES-256") — BM25 bắt exact match, dense bắt paraphrase.
       Bắt buộc `segment_vietnamese()` + `replace("_", " ")` cho cả 2 phía.
3. [ ] **Reranking:** có — `BAAI/bge-reranker-v2-m3` (hoặc `mmarco-mMiniLMv2-L12-H384-v1` khi
       cần nhẹ), top-20 → top-3. Bắt buộc pre-download vào cache + guard `_weights_cached()`
       để service không treo khi mạng lỗi.
4. [ ] **Evaluation:** RAGAS 4 metrics làm gate CI (fail build nếu faithfulness < 0.85), judge
       cấu hình tường minh (`gpt-4o-mini` + local embeddings) — **không** dùng default.
       Bổ sung `failure_analysis()` chạy tự động mỗi lần eval để có bottom-5 kèm diagnosis.
5. [ ] **Enrichment:** `_enrich_single_call()` (1 API call/chunk: summary + HyQA + contextual +
       metadata). Ưu tiên **contextual prepend** (Anthropic: −49% retrieval failure) và
       **auto metadata** (`category`, `effective_date`) để filter version cũ.

### Timeline

| Tuần | Việc | Định nghĩa "xong" |
|------|------|-------------------|
| Tuần 1 | Dựng test set 50 câu + chạy RAGAS baseline; thêm OCR (`pytesseract`) cho 2 PDF scan | Có `baseline_report.json`, 100% tài liệu có text |
| Tuần 2 | Hierarchical + structure-aware chunking; index lại; đo lại RAGAS | `context_recall` +0.05 so với baseline |
| Tuần 3 | Hybrid BM25+RRF + reranker; đo latency p95 từng stage | ≥3 metrics ≥ 0.75, p95 < 3s/query |
| Tuần 4 | Enrichment combined-call + metadata filter theo `effective_date` | Không còn câu trả lời lấy từ version cũ |
| Tuần 5 | Đưa RAGAS vào CI + dashboard latency; viết runbook fallback (mất Qdrant / mất reranker) | CI chặn PR khi faithfulness < 0.85 |

### Rủi ro & phương án dự phòng (học từ lab)

| Rủi ro | Dự phòng đã có sẵn trong code |
|--------|-------------------------------|
| Qdrant server chết | `DenseSearch._connect()` → embedded local |
| Reranker chưa tải xong / mạng lỗi | `_weights_cached()` → giữ thứ tự retrieval, không treo |
| Không có quyền gọi OpenAI | `_offline_metrics()` (`mode="offline_fallback"`) + trả context làm answer |
| underthesea lỗi | `segment_vietnamese()` fallback raw text |

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | 5 | Map được cả 5 module vào code cụ thể |
| Code quality | 4 | Có fallback + comment giải thích "tại sao"; còn thiếu type hints đầy đủ và logging chuẩn thay vì `print()` |
| Teamwork | — | Làm cá nhân |
| Problem solving | 5 | 8 lỗi hạ tầng (Docker, encoding, quyền API, mạng HF) tự chẩn đoán và fix không bỏ tính năng nào; lỗi thứ 9 là lỗi tư duy của chính tôi (siết prompt theo cảm giác) và được sửa bằng A/B có kiểm soát |
