"""Lexical Retriever implementing BM25 ranking on Strategy B chunks."""
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text by words."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"\w+", normalized)


class BM25LexicalRetriever:
    def __init__(
        self,
        chunks_path: Path | str | None = None,
        strategy: str = "B",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.strategy = strategy
        self.k1 = k1
        self.b = b

        if chunks_path is None:
            chunks_path = ROOT / "data/converted/chunks.json"
        self.chunks_path = Path(chunks_path)

        all_chunks = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        if self.strategy:
            self.chunks = [c for c in all_chunks if c["id"].startswith(f"{self.strategy}_")]
        else:
            self.chunks = all_chunks

        self.N = len(self.chunks)
        self.doc_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)

        # Precompute document frequencies and IDFs
        df = Counter(term for tokens in [set(t) for t in self.doc_tokens] for term in tokens)
        self.idf = {
            term: math.log((self.N - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search chunks using Okapi BM25 scoring."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        q_tokens = tokenize(query)
        chapter = (filters or {}).get("chapter")

        scored_results: list[dict[str, Any]] = []
        for c, tokens, doc_len in zip(self.chunks, self.doc_tokens, self.doc_lens):
            if chapter and c.get("metadata", {}).get("chapter") != chapter:
                continue

            tf = Counter(tokens)
            score = 0.0
            for q in q_tokens:
                if q in tf:
                    f = tf[q]
                    numerator = self.idf.get(q, 0.0) * f * (self.k1 + 1.0)
                    denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += numerator / denominator

            if score > 0:
                scored_results.append({
                    "id": c["id"],
                    "source_id": c["source_id"],
                    "source": c["source"],
                    "page": c["page"],
                    "text": c["text"],
                    "metadata": c.get("metadata", {}),
                    "score": float(score),
                })

        # Sort by score descending, breaking ties by chunk id
        scored_results.sort(key=lambda x: (-x["score"], x["id"]))
        return scored_results[:top_k]
