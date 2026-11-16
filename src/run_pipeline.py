"""Pipeline runner for Agentic RAG evaluation and benchmarking."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root and starter to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

from agentic import run_agent
from baseline import hit
from src.dense_retriever import PgVectorDenseRetriever
from src.lexical_retriever import BM25LexicalRetriever
from src.hybrid_retriever import HybridRRFRetriever
from src.llm_controller import OpenRouterController


def evaluate_split(
    split: str,
    method: str = "dense",
    max_searches: int = 3,
    top_k: int = 3,
    budget: int = 800,
    output_dir: Path | None = None,
):
    # Load dataset
    cases_path = ROOT / "data/converted/cases.json"
    questions_path = ROOT / "data/converted/questions.json"

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    questions = json.loads(questions_path.read_text(encoding="utf-8"))

    q_map = {q["id"]: q for q in questions}

    selected_cases = [c for c in cases if c.get("split") == split]
    if not selected_cases:
        raise ValueError(f"No cases found for split '{split}'")

    print(f"\n=======================================================")
    print(f"RUNNING AGENTIC PIPELINE: Split={split.upper()} | Method={method.upper()}")
    print(f"Config: max_searches={max_searches}, top_k={top_k}, budget={budget} chars")
    print(f"Total cases: {len(selected_cases)} (Answerable: {sum(1 for c in selected_cases if c['answerable'])}, Unanswerable: {sum(1 for c in selected_cases if not c['answerable'])})")
    print(f"=======================================================\n")

    # Initialize components
    if method == "dense":
        retriever = PgVectorDenseRetriever()
    elif method == "lexical":
        retriever = BM25LexicalRetriever()
    elif method == "hybrid":
        retriever = HybridRRFRetriever()
    else:
        raise ValueError(f"Unsupported method '{method}'. Choose from dense, lexical, hybrid.")

    controller = OpenRouterController()

    traces = []
    reviews_draft = []

    answerable_count = 0
    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0

    tool_called_count = 0
    provider_errors = 0

    for idx, case in enumerate(selected_cases, 1):
        cid = case["id"]
        q_text = case["question"]
        is_answerable = case["answerable"]
        gold = q_map.get(cid, {}).get("gold", [])

        print(f"[{idx}/{len(selected_cases)}] Case {cid} ({'Answerable' if is_answerable else 'Unanswerable'}): {q_text[:70]}...")

        t_start = time.perf_counter()
        try:
            res = run_agent(
                q_text,
                controller,
                retriever,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            status = res["status"]
            answer = res["answer"]
            citations = res["citations"]
            events = res["events"]

            # Tool use tracking
            has_search = any(e.get("event") == "search" for e in events)
            if has_search:
                tool_called_count += 1

            # Retrieval ranking & context metrics for answerable cases with gold
            case_hit1 = False
            case_hit3 = False
            case_ctx_hit = False

            if is_answerable and gold:
                answerable_count += 1
                # Find first search event
                search_events = [e for e in events if e.get("event") == "search"]
                if search_events:
                    first_results = search_events[0].get("results", [])
                    if first_results:
                        case_hit1 = hit(first_results[0], gold)
                        case_hit3 = any(hit(r, gold) for r in first_results[:3])

                    # Check all context chunks
                    last_context = search_events[-1].get("context", "")
                    last_ctx_ids = search_events[-1].get("context_ids", [])
                    # Gold evidence substring in context
                    case_ctx_hit = any(g["evidence"] in last_context for g in gold)

                if case_hit1:
                    hit1_count += 1
                if case_hit3:
                    hit3_count += 1
                if case_ctx_hit:
                    context_hit_count += 1

            trace_entry = {
                "query_id": cid,
                "split": split,
                "question": q_text,
                "answerable": is_answerable,
                "status": status,
                "answer": answer,
                "citations": citations,
                "events": events,
                "elapsed_ms": elapsed_ms,
                "hit1": case_hit1 if is_answerable else None,
                "hit3": case_hit3 if is_answerable else None,
                "context_hit": case_ctx_hit if is_answerable else None,
            }
            traces.append(trace_entry)

            print(f"    -> Status: {status} | Citations: {citations} | Time: {elapsed_ms:.1f}ms")
            if is_answerable:
                print(f"    -> Hit@1: {case_hit1} | Hit@3: {case_hit3} | ContextHit: {case_ctx_hit}")
            print(f"    -> Answer: {answer[:100]}...\n")

            # Prepare draft review entry for human audit
            if is_answerable:
                # Grade 2 if answered with context hit, 0 if abstain, 1 partial
                initial_grade = 2 if (status == "answered" and case_ctx_hit) else (0 if status == "abstain" else 1)
                reviews_draft.append({
                    "query_id": cid,
                    "answer_grade": initial_grade,
                    "citation_support": [True for _ in citations],  # Default True for review
                    "notes": "Tự động đánh giá sơ bộ, cần đọc đối chiếu."
                })
            else:
                reviews_draft.append({
                    "query_id": cid,
                    "abstention_correct": (status == "abstain"),
                    "citation_support": [],
                    "notes": "Từ chối đúng theo absence_reason." if status == "abstain" else "Lỗi: Đã trả lời trên câu thiếu bằng chứng."
                })

        except Exception as e:
            provider_errors += 1
            print(f"    -> [ERROR] Provider/retriever failure: {e}\n")
            traces.append({
                "query_id": cid,
                "split": split,
                "question": q_text,
                "answerable": is_answerable,
                "status": "error",
                "answer": "",
                "citations": [],
                "events": [{"event": "error", "error": str(e)}],
                "elapsed_ms": 0,
            })

    # Summary calculations
    hit1_rate = (hit1_count / answerable_count) if answerable_count else 0.0
    hit3_rate = (hit3_count / answerable_count) if answerable_count else 0.0
    ctx_rate = (context_hit_count / answerable_count) if answerable_count else 0.0
    tool_rate = (tool_called_count / len(selected_cases)) if selected_cases else 0.0

    summary = {
        "method": method,
        "split": split,
        "total_cases": len(selected_cases),
        "answerable_cases": answerable_count,
        "provider_errors": provider_errors,
        "tool_use": f"{tool_called_count}/{len(selected_cases)} ({tool_rate:.1%})",
        "hit1": f"{hit1_count}/{answerable_count} ({hit1_rate:.1%})",
        "hit3": f"{hit3_count}/{answerable_count} ({hit3_rate:.1%})",
        "context_evidence_rate": f"{context_hit_count}/{answerable_count} ({ctx_rate:.1%})",
        "config": {
            "max_searches": max_searches,
            "top_k": top_k,
            "budget": budget,
        }
    }

    print("\n================== BENCHMARK SUMMARY ==================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=======================================================\n")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "reviews_draft.json").write_text(
            json.dumps(reviews_draft, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[+] Output written to {output_dir}")

    return summary, traces, reviews_draft


def main():
    parser = argparse.ArgumentParser(description="Chạy đánh giá Agentic RAG")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--method", choices=("dense", "lexical", "hybrid"), default="dense")
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out = args.output
    if out is None:
        out = ROOT / f"starter/results/{args.method}_{args.split}_b{args.budget}"

    evaluate_split(
        split=args.split,
        method=args.method,
        max_searches=args.max_searches,
        top_k=args.top_k,
        budget=args.budget,
        output_dir=out,
    )


if __name__ == "__main__":
    main()
