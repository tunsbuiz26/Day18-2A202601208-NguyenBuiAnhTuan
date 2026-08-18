# Failure Analysis — Lab 18: Production RAG

**Nhóm:** làm cá nhân (1 người / toàn bộ 5 modules)
**Thành viên:** Nguyễn Bùi Anh Tuấn — 2A202601208 → M1 + M2 + M3 + M4 + M5 + `pipeline.py`
**Ngày:** 18/08/2026 · **Test set:** 20 câu (`test_set.json`) · **Judge:** `gpt-4o-mini` + local embeddings

---

## RAGAS Scores

Nguồn số: `reports/naive_baseline_report.json` (chunk_basic 500 → dense-only → LLM) và
`reports/ragas_report.json` (production: hierarchical → enrichment → hybrid+RRF → rerank → LLM).

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8000 | 0.7792 | −0.0208 |
| Answer Relevancy | 0.6283 | 0.7110 | **+0.0827** |
| Context Precision | 0.9417 | 0.9750 | +0.0333 |
| Context Recall | 0.8250 | 0.9250 | **+0.1000** |

Ba metric tăng, riêng **faithfulness giảm nhẹ** — không phải nhiễu mà là hệ quả trực tiếp của
pipeline mạnh hơn: naive trả lời cụt hoặc từ chối nhiều (ít claim → dễ "faithful"), production
có đủ context nên trả lời trọn vẹn, nhiều claim hơn → mỗi claim thiếu bằng chứng đều bị trừ.
Nói cách khác faithfulness phạt **số claim**, không phạt **thiếu thông tin** — nên phải đọc nó
cùng answer_relevancy/context_recall, không đọc rời.

**Latency** (`reports/ragas_report.json` → `latency`): retrieval 18.1ms · rerank 1.4ms ·
generation **2120ms** (p95 2971ms) → **99.1% thời gian/query nằm ở LLM**.
Build: chunking 0.12s · enrichment 0.01s (cache hit, lần đầu 411s) · indexing 28.3s.

## A/B answer prompt (cùng index, cùng enrichment cache → chỉ khác prompt)

Đây là thí nghiệm quan trọng nhất của phần optimize; report thô lưu ở `reports/ab_prompt_runs/`.

| Prompt | Faithfulness | Answer Relevancy | Ctx Precision | Ctx Recall | Avg |
|--------|-------------|------------------|---------------|------------|-----|
| v1 — 1 dòng "chỉ dựa trên context" (template) | **0.8125** | 0.6387 | 0.9750 | 0.9250 | 0.8378 |
| v2 — siết: cấm mọi suy luận + bắt buộc từ chối | 0.7333 | 0.6871 | 0.9500 | 0.9000 | 0.8176 |
| v3 — **đang dùng**: nhắc chủ thể, cho tính toán có số gốc | 0.7792 | **0.7110** | 0.9750 | 0.9250 | **0.8476** |
| v4 — v3 + ép trả lời ngắn nhất (1 câu) | 0.7125 | 0.7148 | 0.9750 | 0.9250 | 0.8323 |

Ba kết luận rút ra:

1. **v2 là bài học đắt nhất.** Rule "KHÔNG cộng/trừ số liệu" xung đột với ground truth
   (Q11 `15+3=18`, Q16 phí quá hạn, Q17 `85% × 20tr`) và rule bắt buộc từ chối gây
   **over-refusal**: 2 câu có context đúng vẫn trả "Không tìm thấy" → faithfulness = 0 cho cả 2.
   Siết prompt theo cảm giác mà không đối chiếu ground truth thì làm điểm tệ đi.
2. **Answer_relevancy và faithfulness đánh đổi nhau.** Càng ép trả lời ngắn (v4) thì AR càng lên,
   F càng xuống: câu ngắn ít lặp lại chủ thể → RAGAS sinh lại câu hỏi sát hơn (AR↑), nhưng mất
   phần dẫn số gốc từ context nên claim kết quả trở thành "không truy được nguồn" (F↓).
3. **CP/CR gần như không đổi giữa các prompt** (0.975 / 0.925, có 1 lần 0.95 / 0.90) — đúng như
   định nghĩa: hai metric này chấm *context*, không chấm answer. Chênh 0.025 = 1/20 câu bị judge
   đổi ý → đó chính là **mức nhiễu của LLM-as-judge**, mọi cải thiện < 0.03 không nên tin.

## Bottom-5 Failures

Xếp theo `avg_score` tăng dần, lấy từ `failure_analysis()` trong `src/m4_eval.py`
(`DIAGNOSTIC_TREE` map `min(4 metrics)` → root cause → fix).

### #1 — avg 0.500 · worst: faithfulness 0.00
- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Thông tin lương được phân loại là dữ liệu **Bí mật** (cấp 3), cấm chia sẻ với
  đồng nghiệp; dữ liệu Bí mật phải mã hóa khi truyền, truy cập theo need-to-know.
- **Got:** "Không tìm thấy thông tin trong tài liệu."
- **Scores:** F 0.00 · AR 0.00 · CP 1.00 · CR 1.00 · `n_contexts = 1`
- **Error Tree:** Output sai → Context đúng? **Không đủ** → Retrieval sai? **Đúng một nửa** →
  Query OK? **Có, nhưng là câu multi-hop 2 tài liệu** → lỗi ở bước hợp nhất context.
- **Root cause:** câu hỏi cần **ghép 2 tài liệu**: `ky_luong.md:11` ("Thông tin lương là dữ liệu
  **Bí mật**") + `phan_loai_du_lieu.md` (bảng 4 cấp, Bí mật = cấp 3). Trace thực tế: top-4
  retrieval **đều là child của cùng 1 parent** `phan_loai_du_lieu.md::parent_0`, nên
  `_expand_to_parent()` dedup 3 chunk sau rerank xuống **đúng 1 context** — mất hẳn phía
  `ky_luong.md`. LLM chỉ thấy bảng phân loại, không thấy dòng nào gắn "lương" vào cấp nào →
  từ chối. Đây là **hành vi đúng của LLM** nhưng RAGAS chấm refusal = F 0 và AR 0.
- **Suggested fix:** giữ **đa dạng nguồn** ở tầng context: nếu sau dedup còn < 3 context thì bù
  bằng child chunk của các `source` khác (hoặc thêm MMR/diversity penalty vào RRF, hoặc
  query decomposition cho câu multi-hop). Diagnosis tự động ghi "LLM hallucinating" là **sai** —
  `DIAGNOSTIC_TREE` cần thêm nhánh riêng cho `answer == refusal`.

### #2 — avg 0.662 · worst: faithfulness 0.00
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn 15 ngày; quá hạn 5 ngày, phí 2%/tháng trên 15.000.000 VNĐ =
  300.000 VNĐ/tháng (pro-rata ≈ 50.000 VNĐ cho 5 ngày).
- **Got:** "…quá hạn 5 ngày (20 − 15) sẽ bị phạt 2%/tháng trên 15.000.000 VNĐ, tức là 300.000 VNĐ."
- **Scores:** F 0.00 · AR 0.65 · CP 1.00 · CR 1.00 · `n_contexts = 1`
- **Error Tree:** Output sai → Context đúng? **Đúng (CP=CR=1)** → vậy lỗi ở **generation**.
- **Root cause:** answer đúng nghiệp vụ nhưng mọi con số kết quả (`5 ngày`, `300.000 VNĐ`) là
  **LLM tự tính**, không có literal trong context → RAGAS coi là claim không truy được nguồn.
  Thử ép ngắn ở prompt v4 ("2%/tháng trên 15.000.000 VNĐ, quá hạn 5 ngày → 50.000 VNĐ" — khớp
  ground truth gần như từng chữ) vẫn bị **F = 0**. Kết luận: với câu cần tính toán, faithfulness
  và ground_truth **xung đột cấu trúc**, không prompt nào chữa được.
- **Suggested fix:** tách 2 loại câu — câu tra cứu chấm bằng faithfulness, câu tính toán chấm
  bằng answer_correctness/exact-match số; hoặc buộc LLM xuất riêng phần "trích dẫn" (verbatim)
  và phần "tính toán", chỉ đưa phần trích dẫn cho faithfulness.

### #3 — avg 0.664 · worst: faithfulness 0.50
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG. Nhân viên thử việc không được nghỉ phép năm; muốn nghỉ phải xin nghỉ
  không lương và được trưởng phòng phê duyệt.
- **Got:** "KHÔNG, nhân viên thử việc không được hưởng phép năm theo quy định trong chính sách
  nghỉ phép năm."
- **Scores:** F 0.50 · AR 0.66 · CP 1.00 · CR 0.50 · `n_contexts = 2`
- **Error Tree:** Output đúng ý nhưng thiếu → Context đúng? **Thiếu (CR = 0.5)** → Retrieval bỏ
  sót chunk chứa phần "phải xin nghỉ không lương, trưởng phòng phê duyệt".
- **Root cause:** vế 2 của ground truth nằm ở chunk khác không vào top-3. Mệnh đề
  "theo quy định trong chính sách nghỉ phép năm" là **claim thừa** LLM tự thêm → mất 0.5 F.
- **Suggested fix:** tăng `RETRIEVAL_TOP_K` trước rerank (hiện 20 → 30) và bỏ rule cho phép
  viện dẫn tên chính sách khi context không nêu; hoặc parent lớn hơn cho document ngắn.

### #4 — avg 0.706 · worst: faithfulness 0.25
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa
  học. Phải hoàn trả bao nhiêu?
- **Expected:** Cam kết làm việc ≥ 1 năm; nghỉ sau 8 tháng là trước hạn → hoàn trả 100% =
  25.000.000 VNĐ.
- **Got:** "…phải hoàn trả 100% chi phí đào tạo đã được tài trợ, tức là 25.000.000 VNĐ."
- **Scores:** F 0.25 · AR 0.57 · CP 1.00 · CR 1.00 · `n_contexts = 2`
- **Error Tree:** Output sai → Context đúng (CP=CR=1) → lỗi **generation**, cùng họ với #2.
- **Root cause:** con số `25.000.000 VNĐ` đến từ **câu hỏi**, không từ context; context chỉ có
  quy tắc "hoàn trả 100% nếu nghỉ trước 1 năm". RAGAS không phân biệt "số lấy từ câu hỏi" và
  "số bịa" → 3/4 claim bị loại.
- **Suggested fix:** prompt yêu cầu diễn đạt theo **tỷ lệ trong context** ("hoàn trả 100% chi phí
  được tài trợ") và chỉ nhắc lại số của người hỏi ở dạng dẫn lại ("với 25 triệu bạn nêu"),
  để claim chính vẫn neo vào context.

### #5 — avg 0.753 · worst: context_precision 0.50
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo v2024 hiện hành: từ 3 năm, cộng 1 ngày cho mỗi 3 năm (v2023 cũ: 5 năm).
- **Got:** "Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi
  3 năm làm việc liên tục theo chính sách nghỉ phép năm phiên bản 2024." (F 1.00 — answer đúng)
- **Scores:** F 1.00 · AR 0.51 · CP 0.50 · CR 1.00 · `n_contexts = 2`
- **Error Tree:** Output đúng → Context có rác? **Có** → 1 trong 2 context là
  `nghi_phep_nam_v2023.md` (phiên bản đã hết hiệu lực) → judge chấm chunk đó irrelevant → CP 0.5.
- **Root cause:** corpus có **2 phiên bản cùng chủ đề** (`v2023` vs `v2024`, tương tự
  `mat_khau_v1` vs `v2`) và pipeline **không có metadata `effective_date`** để lọc.
- **Suggested fix:** M5 `extract_metadata()` đã trả `category/topic`; bổ sung `effective_date` +
  `version`, rồi filter ở tầng Qdrant (`query_points(query_filter=...)`) hoặc hạ điểm chunk cũ
  trước RRF. Đây cũng là fix cho AR: answer không phải kể cả 2 phiên bản nữa.

## Case Study (cho presentation)

**Question chọn phân tích:** *"Thông tin lương thuộc cấp độ phân loại dữ liệu nào?"* — chọn vì nó
là câu duy nhất bị **2 metric = 0** cùng lúc (F 0.00 + AR 0.00), một mình nó kéo faithfulness
tổng xuống ~0.05 và answer_relevancy xuống ~0.036 trên 20 câu.

**Trace thật (chạy lại đúng query này qua pipeline):**

```
RETRIEVED 20 (hybrid BM25+Dense, RRF k=60) — top-4:
  0.0328  phan_loai_du_lieu.md   pk=phan_loai_du_lieu.md::parent_0
  0.0323  phan_loai_du_lieu.md   pk=phan_loai_du_lieu.md::parent_0
  0.0317  phan_loai_du_lieu.md   pk=phan_loai_du_lieu.md::parent_0
  0.0303  phan_loai_du_lieu.md   pk=phan_loai_du_lieu.md::parent_0
  0.0293  so_tay_an_toan.pdf     0.0289  danh_gia_hieu_suat.md
RERANKED top-3 → cả 3 đều pk=phan_loai_du_lieu.md::parent_0
_expand_to_parent() → FINAL CONTEXTS = 1  (908 ký tự, bảng 4 cấp phân loại)
→ trong context có chữ "Bí mật" nhưng KHÔNG có chữ "lương"
```

**Error Tree walkthrough:**

1. **Output đúng?** Không — "Không tìm thấy thông tin trong tài liệu.", trong khi tài liệu có.
2. **Context đúng?** Đúng *một nửa*. CP 1.0 / CR 1.0 nói "context liên quan và phủ ground truth",
   nhưng đó là judge chấm rộng tay: context có bảng phân loại (đủ để khớp phần lớn câu ground
   truth) mà **thiếu mắt nối** `ky_luong.md:11` — "Thông tin lương là dữ liệu **Bí mật**".
   → Bài học: **CP/CR cao vẫn có thể fail** với câu multi-hop; đừng tin CP/CR một mình.
3. **Query rewrite OK?** Query không sai chính tả, không thiếu từ khóa. Nhưng nó là **multi-hop**:
   "thông tin lương" (doc A) + "cấp độ phân loại" (doc B). Cụm "phân loại dữ liệu" trùng gần như
   nguyên văn tên doc B nên **BM25 và dense đồng thuận đẩy 4 child của doc B lên top**, đúng
   trường hợp hybrid *không* cứu được nhau vì cả hai cùng lệch một hướng.
4. **Fix ở bước nào:** không phải retrieval và không phải LLM — mà ở **`_expand_to_parent()`**:
   dedup theo parent làm 3 chunk (đa dạng vị trí) co lại thành **1 context, 1 nguồn**. Small-to-big
   giúp context_recall +0.125 ở các câu single-hop, nhưng chính nó **giết diversity** ở câu
   multi-hop. Fix rẻ nhất: sau dedup nếu còn < 3 context thì bù thêm child chunk từ `source` khác
   (giữ nguyên parent cho những cái đã có), tức đảm bảo ràng buộc "≥ 2 nguồn khi câu hỏi
   nhắc ≥ 2 thực thể".

**Nếu có thêm 1 giờ, sẽ optimize (thứ tự theo điểm/giờ):**

- **(20 phút) Diversity ở tầng context:** bù child chunk khác `source` khi dedup parent làm rơi
  context xuống < 3, hoặc thêm penalty cho chunk cùng parent trong RRF. Sửa được #1 → kỳ vọng
  F +0.05, AR +0.036 chỉ với 1 câu.
- **(20 phút) `effective_date` / `version` vào auto-metadata + filter Qdrant:** sửa #5 và câu
  "đổi mật khẩu" (CP 0.5 do chunk `mat_khau_v1`) → CP về ~1.0, AR đỡ bị trừ vì answer không phải
  kể 2 phiên bản.
- **(10 phút) Sửa `DIAGNOSTIC_TREE`:** thêm nhánh `answer là refusal` → "context thiếu mắt nối,
  không phải hallucination". Hiện tại #1 bị chẩn đoán ngược, gây mất thời gian debug sai chỗ.
- **(10 phút) Tách metric theo loại câu hỏi:** câu tính toán (#2, #4, Q11, Q17) chấm bằng
  answer_correctness thay vì faithfulness — hiện chúng đóng góp gần hết phần faithfulness bị mất
  mà không có cách nào sửa bằng prompt (đã chứng minh bằng A/B v1→v4).
- **(nếu tải được model) Bật cross-encoder thật:** cả 5 run đều chạy nhánh fallback
  (`giữ thứ tự retrieval`), nên toàn bộ điểm hiện tại là **retrieval-only** — rerank thật rất có
  thể tự sửa #1 và #5 vì nó chấm từng cặp (query, doc) thay vì tin RRF.
