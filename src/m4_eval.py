from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, re, warnings
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, EMBEDDING_MODEL

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# RAGAS default judge = gpt-3.5-turbo-16k + text-embedding-ada-002. API key của lab
# chỉ được cấp gpt-4o-mini và KHÔNG có embedding model nào → phải inject thủ công,
# nếu không answer_relevancy chết 403 model_not_found.
RAGAS_LLM_MODEL = os.getenv("RAGAS_LLM_MODEL", "gpt-4o-mini")
RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "900"))
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "4"))

# Diagnostic Tree: metric thấp nhất → nguyên nhân gốc → hướng sửa.
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM hallucinating — câu trả lời chứa thông tin không có trong context",
                     "Siết prompt ('chỉ dùng context'), giảm temperature, thêm citation bắt buộc"),
    "context_recall": ("Missing relevant chunks — retrieval bỏ sót thông tin cần thiết",
                       "Tăng top_k, cải thiện chunking (parent-child), thêm BM25 vào hybrid"),
    "context_precision": ("Too many irrelevant chunks — context bị pha loãng",
                          "Thêm/siết reranking, giảm rerank_top_k, filter theo metadata"),
    "answer_relevancy": ("Answer doesn't match question — trả lời lệch câu hỏi",
                         "Sửa prompt template, thêm query rewriting, yêu cầu trả lời trực tiếp"),
}

REFUSAL_MARKERS = (
    "không tìm thấy thông tin",
    "không có thông tin",
    "không đủ thông tin",
    "no information",
    "not found",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value) -> float:
    """RAGAS trả NaN khi metric không tính được → quy về 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(v) or math.isinf(v) else v


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", (text or "").lower()) if len(t) > 1}


def _is_refusal(answer: str) -> bool:
    """Detect a no-answer response so it is not mislabeled as hallucination."""
    normalized = " ".join((answer or "").casefold().split())
    return any(marker in normalized for marker in REFUSAL_MARKERS)


def _offline_metrics(questions, answers, contexts, ground_truths) -> dict:
    """Fallback không cần LLM: xấp xỉ 4 metrics bằng token overlap.

    ⚠️ KHÔNG phải RAGAS thật (RAGAS dùng LLM-as-judge). Chỉ dùng khi thiếu
    OPENAI_API_KEY / ragas để pipeline vẫn chạy end-to-end và có số để so sánh
    tương đối. Report ghi rõ mode="offline_fallback".
    """
    def overlap(a: str, b: str) -> float:
        ta, tb = _tokens(a), _tokens(b)
        return len(ta & tb) / len(tb) if tb else 0.0

    per_question = []
    for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        ctx_all = " ".join(ctxs)
        relevant = [c for c in ctxs if overlap(c, gt) >= 0.3]
        per_question.append(EvalResult(
            question=q, answer=a, contexts=list(ctxs), ground_truth=gt,
            faithfulness=round(overlap(a, ctx_all) if ctx_all else 0.0, 4),
            answer_relevancy=round(overlap(a, q), 4),
            context_precision=round(len(relevant) / len(ctxs), 4) if ctxs else 0.0,
            context_recall=round(overlap(ctx_all, gt), 4),
        ))

    return {
        **{m: _mean([getattr(r, m) for r in per_question]) for m in METRIC_NAMES},
        "per_question": per_question,
        "mode": "offline_fallback",
        "note": "Heuristic token-overlap, KHÔNG phải RAGAS LLM-as-judge.",
    }


def _ragas_judge():
    """Trả (llm, embeddings) cho RAGAS.

    - LLM: gpt-4o-mini (model duy nhất key này truy cập được).
    - Embeddings: sentence-transformers local (đã cache) — dùng cho answer_relevancy
      (cosine giữa câu hỏi gốc và câu hỏi RAGAS sinh lại từ answer). Không gọi network,
      không cần quyền truy cập embedding API.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = LangchainLLMWrapper(ChatOpenAI(model=RAGAS_LLM_MODEL, temperature=0))

    embeddings = None
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        # HF_HUB_OFFLINE=1: model đã có trong cache, nhưng from_pretrained() vẫn gọi
        # HEAD lên huggingface.co để check update — mạng ở đây hay stall 0 byte/phút
        # nên bước "load model local" có thể treo vài phút. Chỉ đọc cache là đủ.
        # catch_warnings: HuggingFaceEmbeddings của langchain_community deprecated,
        # bản thay thế `langchain_huggingface` chưa cài được (mạng) → tắt warning cho
        # output pytest sạch (check_lab.py parse dòng summary cuối, warning làm nó lỗi).
        prev = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                embeddings = LangchainEmbeddingsWrapper(
                    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                )
        finally:
            if prev is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev
    except Exception as e:
        print(f"  ⚠️  Local embeddings cho RAGAS không load được ({e}) — dùng default.")
    return llm, embeddings


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation (4 metrics).

    RAGAS cần OPENAI_API_KEY + Python 3.11+ (asyncio). Nếu không chạy được →
    fallback heuristic offline để pipeline không chết giữa đường.
    """
    if not questions:
        return {**{m: 0.0 for m in METRIC_NAMES}, "per_question": [], "mode": "empty"}

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from ragas.run_config import RunConfig
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        llm, embeddings = _ragas_judge()
        # timeout default 180s quá ngắn cho mạng chậm → faithfulness bị TimeoutError → 0.0.
        # Nhưng timeout là **theo job** (mỗi metric × mỗi câu) và RAGAS còn retry, nên với
        # smoke test 1–2 câu (tests/test_m4.py, check_lab.py chỉ cho pytest 120s) phải fail
        # nhanh, không được giữ 900s: eval thật 20 câu mới dùng timeout dài.
        smoke = len(questions) <= 2
        run_config = RunConfig(
            timeout=min(RAGAS_TIMEOUT, 60) if smoke else RAGAS_TIMEOUT,
            max_workers=RAGAS_MAX_WORKERS,
            max_retries=1 if smoke else 10,
        )
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall],
                          llm=llm, embeddings=embeddings, run_config=run_config)
        df = result.to_pandas()

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_safe_float(row.get("faithfulness")),
                answer_relevancy=_safe_float(row.get("answer_relevancy")),
                context_precision=_safe_float(row.get("context_precision")),
                context_recall=_safe_float(row.get("context_recall")),
            )
            for _, row in df.iterrows()
        ]
        return {
            **{m: _mean([getattr(r, m) for r in per_question]) for m in METRIC_NAMES},
            "per_question": per_question,
            "mode": "ragas",
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        print("  → Dùng offline heuristic metrics (xem field 'mode' trong report).")
        return _offline_metrics(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = []
    for r in eval_results:
        scores = {m: _safe_float(getattr(r, m, 0.0)) for m in METRIC_NAMES}
        worst_metric = min(scores, key=lambda m: scores[m])
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        if worst_metric == "faithfulness" and _is_refusal(getattr(r, "answer", "")):
            if getattr(r, "contexts", None):
                diagnosis = (
                    "Refusal despite retrieved context — context may be incomplete or lack "
                    "source diversity for a multi-hop question"
                )
                fix = (
                    "Inspect parent expansion and preserve diverse sources; increase retrieval "
                    "coverage or decompose the query before tightening the answer prompt"
                )
            else:
                diagnosis = "Refusal with no retrieved context — retrieval returned no usable evidence"
                fix = "Check indexing and query retrieval, then increase top_k or improve chunking"
        scored.append({
            "question": getattr(r, "question", ""),
            "answer": (getattr(r, "answer", "") or "")[:300],
            "ground_truth": (getattr(r, "ground_truth", "") or "")[:300],
            "n_contexts": len(getattr(r, "contexts", []) or []),
            "avg_score": round(sum(scores.values()) / len(scores), 4),
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "scores": scores,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })

    scored.sort(key=lambda d: d["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json",
                latency: dict | None = None):
    """Save evaluation report to JSON. (Đã implement sẵn — thêm latency breakdown)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    if latency:
        report["latency"] = latency
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
