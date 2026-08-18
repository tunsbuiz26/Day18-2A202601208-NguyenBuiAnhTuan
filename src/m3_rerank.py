from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K, RERANKER_MODEL

# Cross-encoder ~2.2GB → cache theo tên model, tránh load lại mỗi lần khởi tạo.
_MODEL_CACHE: dict[str, object] = {}
# Model chưa tải đủ → chỉ cảnh báo 1 lần thay vì mỗi query.
_WARNED: set[str] = set()

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_REQUIRED_MODEL_FILES = ("modules.json",)


def _weights_cached(model_name: str) -> bool:
    """Weights đã có sẵn trong HF cache chưa?

    Bắt buộc kiểm tra trước khi gọi CrossEncoder(): khi mạng tới huggingface.co bị
    stall (0 byte/phút), from_pretrained() retry vô hạn và treo cả pipeline —
    thà rerank fallback còn hơn treo.
    """
    if os.path.isdir(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return True  # không xác định được → cứ thử load
    has_weights = any(
        isinstance(try_to_load_from_cache(model_name, fn), str)
        for fn in _WEIGHT_FILES
    )
    has_structure = all(
        isinstance(try_to_load_from_cache(model_name, fn), str)
        for fn in _REQUIRED_MODEL_FILES
    )
    return has_weights and has_structure


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        # False is a cached "load failed" state. It prevents every query from
        # retrying a broken/incomplete Hugging Face cache.
        if self._model is False:
            return None
        if self._model is None:
            self._model = _MODEL_CACHE.get(self.model_name)
            if self._model is not None:
                return self._model
            if not _weights_cached(self.model_name):
                if self.model_name not in _WARNED:
                    _WARNED.add(self.model_name)
                    print(f"  ⚠️  Reranker '{self.model_name}' chưa có trong HF cache "
                          f"(mạng không tải nổi) — bỏ qua rerank, giữ thứ tự retrieval.")
                return None
            # Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding:
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            # HF_HUB_OFFLINE=1: chỉ đọc cache, không để from_pretrained treo vì network.
            try:
                from sentence_transformers import CrossEncoder
                prev = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "1"
                try:
                    self._model = CrossEncoder(self.model_name, local_files_only=True)
                finally:
                    if prev is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = prev
            except Exception as e:
                self._model = False
                if self.model_name not in _WARNED:
                    _WARNED.add(self.model_name)
                    print(f"  ⚠️  Không load được reranker '{self.model_name}' ({e}) "
                          "— dùng thứ tự retrieval làm fallback.")
                return None
            _MODEL_CACHE[self.model_name] = self._model
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        try:
            model = self._load_model()
            if model is None:
                raise RuntimeError("reranker unavailable")
            pairs = [(query, doc.get("text", "")) for doc in documents]
            scores = model.predict(pairs)
        except Exception as e:
            if str(e) != "reranker unavailable":
                print(f"  ⚠️  Rerank failed ({e}) — giữ nguyên thứ tự retrieval.")
            scores = [doc.get("score", 0.0) for doc in documents]

        if isinstance(scores, (int, float)):
            scores = [scores]

        # Cross-encoder chấm từng cặp (query, doc) nên xếp hạng chính xác hơn
        # bi-encoder, nhưng phải chạy N lần forward → chỉ dùng ở top-N nhỏ.
        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = _MODEL_CACHE.get("flashrank")
            if self._model is None:
                self._model = Ranker()
                _MODEL_CACHE["flashrank"] = self._model
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import RerankRequest
            model = self._load_model()
            passages = [{"id": i, "text": d.get("text", ""), "meta": d.get("metadata", {})}
                        for i, d in enumerate(documents)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
        except Exception as e:
            print(f"  ⚠️  Flashrank failed: {e}")
            return []
        return [
            RerankResult(
                text=r["text"],
                original_score=float(documents[r["id"]].get("score", 0.0)),
                rerank_score=float(r["score"]),
                metadata=r.get("meta", {}),
                rank=i,
            )
            for i, r in enumerate(results[:top_k])
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
