from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib, os, re, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
                    QDRANT_LOCAL_PATH, QDRANT_TIMEOUT)

_qdrant_clients: dict[str, object] = {}   # path/url → client (local mode chỉ cho 1 handle/process)
_encoders: dict[str, object] = {}         # model_name → SentenceTransformer (1 lần/process)


def _load_encoder(model_name: str):
    """Load embedding model từ HF cache, không cho phép chạm network.

    Cùng lý do như `m3_rerank._load_model()`: model đã cache nhưng
    `from_pretrained()` vẫn gọi HEAD lên huggingface.co để check update, và mạng ở
    đây hay stall 0 byte/phút → bước "load model local" treo vài phút, kéo cả
    pytest/pipeline theo. `HF_HUB_OFFLINE=1` khiến nó chỉ đọc cache.
    """
    if model_name in _encoders:
        return _encoders[model_name]
    prev_hf = os.environ.get("HF_HUB_OFFLINE")
    prev_transformers = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        # Set offline flags before importing transformers; several versions
        # snapshot the flags at import time.
        from sentence_transformers import SentenceTransformer
        _encoders[model_name] = SentenceTransformer(model_name, local_files_only=True)
    finally:
        if prev_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_hf
        if prev_transformers is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_transformers
    return _encoders[model_name]


class _HashEncoder:
    """Deterministic local fallback when a Hugging Face model is unavailable.

    This is deliberately not presented as a semantic model. It keeps the
    Qdrant/DenseSearch path operational offline while BM25 remains the main
    source of lexical relevance until the configured model is downloaded.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

    def encode(self, sentences, show_progress_bar: bool = False):
        import numpy as np

        single = isinstance(sentences, str)
        items = [sentences] if single else list(sentences)
        vectors = np.zeros((len(items), self.dimension), dtype=np.float32)
        for row, sentence in enumerate(items):
            tokens = re.findall(r"\w+", str(sentence).casefold())
            for token in tokens:
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vectors[row, index] += sign
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors[0] if single else vectors


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words.

    underthesea nối từ ghép bằng "_" ("nghỉ_phép"). BM25 tokenize bằng split(" ")
    nên phải replace("_", " ") — nếu không, query "nghỉ phép" (2 token) sẽ không
    khớp document token "nghỉ_phép".
    """
    if not text or not text.strip():
        return text
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception as e:
        print(f"  ⚠️  underthesea segmentation failed ({e}) — dùng raw text.")
        return text  # fallback


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(c.get("text", "")).lower().split()
            for c in chunks
        ]
        if not any(self.corpus_tokens):
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i].get("text", ""),
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices if scores[i] > 0   # bỏ doc không match token nào
        ]


class DenseSearch:
    def __init__(self):
        self.client, self.mode = self._connect()
        self._encoder = None
    @staticmethod
    def _connect():
        """Ưu tiên Qdrant server (Docker); không có thì dùng embedded local storage.

        Client được cache theo target vì local mode giữ file lock — mở 2 client
        trên cùng thư mục trong 1 process sẽ lỗi "already accessed by another instance".
        """
        from qdrant_client import QdrantClient

        url = f"{QDRANT_HOST}:{QDRANT_PORT}"
        if url in _qdrant_clients:
            return _qdrant_clients[url], "server"
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=QDRANT_TIMEOUT)
            client.get_collections()          # ping — QdrantClient không connect lúc init
            _qdrant_clients[url] = client
            return client, "server"
        except Exception:
            if QDRANT_LOCAL_PATH not in _qdrant_clients:
                print(f"  ⚠️  Không kết nối Qdrant {url} — dùng embedded local: {QDRANT_LOCAL_PATH}")
                _qdrant_clients[QDRANT_LOCAL_PATH] = QdrantClient(path=QDRANT_LOCAL_PATH)
            return _qdrant_clients[QDRANT_LOCAL_PATH], "local"

    def _get_encoder(self):
        if self._encoder is None:
            try:
                self._encoder = _load_encoder(EMBEDDING_MODEL)
            except Exception as e:
                print(f"  ⚠️  Không load được embedding '{EMBEDDING_MODEL}' ({e}) "
                      f"— dùng hash fallback {EMBEDDING_DIM} chiều.")
                self._encoder = _HashEncoder(EMBEDDING_DIM)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        if not chunks:
            return

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [c.get("text", "") for c in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={**chunks[i].get("metadata", {}), "text": texts[i]},
            )
            for i in range(len(chunks))
        ]
        self.client.upsert(collection, points)

    def search(self, query: str, top_k: int = DENSE_TOP_K,
               collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # qdrant-client >= 1.10 dùng query_points(), search() đã bị bỏ.
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            response = self.client.query_points(collection, query=query_vector, limit=top_k)
        except Exception as e:
            print(f"  ⚠️  Dense search failed: {e}")
            return []
        return [
            SearchResult(
                text=pt.payload.get("text", ""),
                score=float(pt.score),
                metadata=pt.payload,
                method="dense",
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank).

    RRF chỉ dùng thứ hạng nên không cần normalize score giữa BM25 (không chặn trên)
    và cosine similarity ([-1, 1]).
    """
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.values(), key=lambda e: e["score"], reverse=True)
    return [
        SearchResult(
            text=e["result"].text,
            score=e["score"],
            metadata=e["result"].metadata,
            method="hybrid",
        )
        for e in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
