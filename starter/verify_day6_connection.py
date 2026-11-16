"""Verify connection to Day 6 PostgreSQL + pgvector database."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from pgvector.psycopg import register_vector

# Load .env
env_path = Path(__file__).resolve().parent / '.env'
if not env_path.exists():
    env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab")


def main():
    print(f"[*] Connecting to PostgreSQL vector store at {DATABASE_URL} ...")
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                # 1. Check table existence
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'chunks'
                    );
                """)
                exists = cur.fetchone()[0]
                if not exists:
                    print("[!] Table 'chunks' does not exist in database!")
                    sys.exit(1)
                print("[+] Table 'chunks' found.")

                # 2. Check counts by strategy
                cur.execute("SELECT strategy, count(*) FROM chunks GROUP BY strategy ORDER BY strategy;")
                counts = cur.fetchall()
                print(f"[+] Chunk counts by strategy: {dict(counts)}")

                # 3. Check vector dimensions
                cur.execute("SELECT chunk_id, strategy, embedding FROM chunks WHERE embedding IS NOT NULL LIMIT 1;")
                sample = cur.fetchone()
                if sample:
                    chunk_id, strat, vec = sample
                    dim = len(vec.to_numpy())
                    print(f"[+] Sample chunk '{chunk_id}' (strategy {strat}) has vector dimension: {dim}")
                else:
                    print("[!] No embeddings found in 'chunks' table!")
                    sys.exit(1)

                # 4. Test Cosine Similarity Query
                cur.execute("""
                    SELECT chunk_id, strategy, 1 - (embedding <=> %s) AS similarity
                    FROM chunks
                    WHERE strategy = 'B'
                    ORDER BY embedding <=> %s
                    LIMIT 3;
                """, (vec, vec))
                results = cur.fetchall()
                print(f"[+] Test Cosine Similarity query returned {len(results)} hits:")
                for r in results:
                    print(f"    - Chunk {r[0]}: similarity = {r[2]:.4f}")

        print("\n[SUCCESS] Day 6 PostgreSQL + pgvector connection is 100% verified and operational!")
    except Exception as e:
        print(f"[ERROR] Failed to connect or query database: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
