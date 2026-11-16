"""Hybrid Retriever combining Dense (Vector) and Lexical (BM25) via Reciprocal Rank Fusion (RRF)."""
from typing import Any
from src.dense_retriever import PgVectorDenseRetriever
from src.lexical_retriever import BM25LexicalRetriever


class HybridRRFRetriever:
    def __init__(
        self,
        dense_retriever: Any | None = None,
        lexical_retriever: Any | None = None,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever or PgVectorDenseRetriever()
        self.lexical_retriever = lexical_retriever or BM25LexicalRetriever()
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Combine dense and lexical results using Reciprocal Rank Fusion."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        # Step 1: Retrieve top candidate chunks from each retriever
        dense_candidates = self.dense_retriever.search(
            query, filters=filters, top_k=self.candidate_k
        )
        lexical_candidates = self.lexical_retriever.search(
            query, filters=filters, top_k=self.candidate_k
        )

        # Step 2: Reciprocal Rank Fusion (RRF)
        chunk_map: dict[str, dict[str, Any]] = {}
        rrf_scores: dict[str, float] = {}

        for rank, item in enumerate(dense_candidates, 1):
            cid = item["id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(lexical_candidates, 1):
            cid = item["id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        # Step 3: Sort by fused RRF score descending
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: (-rrf_scores[cid], cid),
        )

        # Step 4: Construct top-k output
        results: list[dict[str, Any]] = []
        for cid in sorted_ids[:top_k]:
            base = dict(chunk_map[cid])
            base["score"] = float(rrf_scores[cid])
            results.append(base)

        return results
