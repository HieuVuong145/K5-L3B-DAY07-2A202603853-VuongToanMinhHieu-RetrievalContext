"""Benchmark script comparing multiple retrieval methods and configurations on Dev/Test."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit, pack_context
from src.dense_retriever import PgVectorDenseRetriever
from src.lexical_retriever import BM25LexicalRetriever
from src.hybrid_retriever import HybridRRFRetriever


def run_benchmark(split: str = "dev"):
    questions_path = ROOT / "data/converted/questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    # Only answerable questions go into the retrieval metric denominator (per BENCHMARK.md)
    selected_questions = [q for q in questions if q.get("split") == split]
    N = len(selected_questions)
    if N == 0:
        raise ValueError(f"No questions found for split '{split}'")

    print(f"\n=======================================================")
    print(f"BENCHMARKING RETRIEVAL METHODS ON SPLIT: {split.upper()} (N={N})")
    print(f"=======================================================\n")

    # Initialize shared retrievers
    dense = PgVectorDenseRetriever()
    lexical = BM25LexicalRetriever()
    hybrid = HybridRRFRetriever(dense_retriever=dense, lexical_retriever=lexical)

    configs = [
        {
            "name": "Baseline (Dense)",
            "method": "dense",
            "retriever": dense,
            "top_k": 3,
            "budget": 800,
            "desc": "Dense retrieval (pgvector + text-embedding-3-small, top_k=3, budget=800)",
            "est_cost_per_q": 0.0000006,  # 30 tokens * $0.00002/1k
        },
        {
            "name": "Phương pháp 2 (Lexical BM25)",
            "method": "lexical",
            "retriever": lexical,
            "top_k": 3,
            "budget": 800,
            "desc": "Lexical BM25 (top_k=3, budget=800)",
            "est_cost_per_q": 0.0,
        },
        {
            "name": "Phương pháp 3 (Hybrid RRF)",
            "method": "hybrid",
            "retriever": hybrid,
            "top_k": 3,
            "budget": 800,
            "desc": "Hybrid RRF (Dense + BM25, candidate_k=10, rrf_k=60, top_k=3, budget=800)",
            "est_cost_per_q": 0.0000006,
        },
        {
            "name": "Cải thiện 1 (Dense + Budget 2000)",
            "method": "dense",
            "retriever": dense,
            "top_k": 3,
            "budget": 2000,
            "desc": "Dense retrieval với context budget mở rộng (800 -> 2000 chars)",
            "est_cost_per_q": 0.0000006,
        },
        {
            "name": "Cải thiện 2 (Hybrid RRF + Budget 2000)",
            "method": "hybrid",
            "retriever": hybrid,
            "top_k": 3,
            "budget": 2000,
            "desc": "Hybrid RRF với context budget mở rộng (800 -> 2000 chars)",
            "est_cost_per_q": 0.0000006,
        },
    ]

    benchmark_rows = []

    for cfg in configs:
        name = cfg["name"]
        retriever = cfg["retriever"]
        top_k = cfg["top_k"]
        budget = cfg["budget"]

        hit1_count = 0
        hit3_count = 0
        ctx_count = 0
        latencies_ms = []

        for q in selected_questions:
            q_text = q["question"]
            gold = q["gold"]

            t0 = time.perf_counter()
            results = retriever.search(q_text, top_k=top_k)
            t_elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies_ms.append(t_elapsed_ms)

            # Hit@1
            c_hit1 = bool(results and hit(results[0], gold))
            if c_hit1:
                hit1_count += 1

            # Hit@3
            c_hit3 = any(hit(r, gold) for r in results[:3])
            if c_hit3:
                hit3_count += 1

            # Context Evidence Rate
            ctx, ids = pack_context(results, budget=budget)
            c_ctx = any(g["evidence"] in ctx for g in gold)
            if c_ctx:
                ctx_count += 1

        median_ms = statistics.median(latencies_ms)
        mean_ms = statistics.mean(latencies_ms)
        total_cost_usd = cfg["est_cost_per_q"] * N

        row = {
            "config_name": name,
            "description": cfg["desc"],
            "split": split,
            "N": N,
            "hit1": f"{hit1_count}/{N} ({hit1_count/N:.1%})",
            "hit3": f"{hit3_count}/{N} ({hit3_count/N:.1%})",
            "context_evidence_rate": f"{ctx_count}/{N} ({ctx_count/N:.1%})",
            "hit1_val": hit1_count / N,
            "hit3_val": hit3_count / N,
            "ctx_val": ctx_count / N,
            "median_ms": f"{median_ms:.1f}ms",
            "mean_ms": f"{mean_ms:.1f}ms",
            "cost_usd": f"${total_cost_usd:.6f}",
        }
        benchmark_rows.append(row)

        print(f"[{name}]")
        print(f"  Hit@1: {row['hit1']} | Hit@3: {row['hit3']} | ContextRate: {row['context_evidence_rate']}")
        print(f"  Latency: median={row['median_ms']}, mean={row['mean_ms']} | Cost: {row['cost_usd']}\n")

    # Output table markdown
    print("\n================== SUMMARY MARKDOWN TABLE ==================")
    print("| Cấu hình | Mô tả / Thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (median) | Chi phí |")
    print("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in benchmark_rows:
        print(f"| {r['config_name']} | {r['description']} | {r['split']} | {r['hit1']} | {r['hit3']} | {r['context_evidence_rate']} | {r['median_ms']} | {r['cost_usd']} |")
    print("============================================================\n")

    # Save to file
    out_dir = ROOT / "starter/results/retrieval_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"benchmark_{split}.json").write_text(
        json.dumps(benchmark_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[+] Benchmark results saved to {out_dir / f'benchmark_{split}.json'}")
    return benchmark_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    run_benchmark(split=args.split)
