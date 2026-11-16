"""Dense Retriever using PostgreSQL + pgvector and OpenRouter embeddings."""
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import psycopg
except ImportError:
    psycopg = None


class PgVectorDenseRetriever:
    def __init__(
        self,
        db_url: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
        default_strategy: str = "B",
    ):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL",
            "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab",
        )
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
        self.default_strategy = default_strategy
        self.cache_file = Path(__file__).resolve().parents[1] / "data/embedding_cache.json"
        self._cache: dict[str, list[float]] = {}
        if self.cache_file.exists():
            try:
                self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def get_embedding(self, text: str) -> list[float]:
        """Fetch embedding from OpenRouter with persistent disk caching."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": self.embedding_model,
                "input": text,
            }).encode("utf-8"),
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                emb = data["data"][0]["embedding"]
                self._cache[text] = emb
                try:
                    self.cache_file.write_text(
                        json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
                    )
                except Exception:
                    pass
                return emb
        except Exception as e:
            raise ConnectionError(f"OpenRouter embedding error: {e}") from e

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Query pgvector for top-k closest chunks using cosine distance."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if psycopg is None:
            raise ConnectionError("psycopg package is not installed.")

        emb = self.get_embedding(query.strip())
        vec_str = str(emb)

        strategy = (filters or {}).get("strategy", self.default_strategy)
        chapter = (filters or {}).get("chapter")

        conditions = []
        params = [vec_str]

        if strategy:
            conditions.append("strategy = %s")
            params.append(strategy)

        if chapter:
            conditions.append("chapter = %s")
            params.append(chapter)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT chunk_id, document_id, source, page, text, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM chunks
            {where_clause}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
        """
        params.extend([vec_str, top_k])

        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
        except Exception as e:
            raise ConnectionError(f"Database pgvector query error: {e}") from e

        results: list[dict[str, Any]] = []
        seen_ids = set()
        for r in rows:
            cid = r[0]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            results.append({
                "id": cid,
                "source_id": r[1],
                "source": r[2],
                "page": r[3],
                "text": r[4],
                "metadata": r[5] if isinstance(r[5], dict) else {},
                "score": float(r[6]),
            })

        return results[:top_k]
