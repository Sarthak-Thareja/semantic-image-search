"""Evaluate retrieval metrics: Recall@K, mAP."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def compute_recall_at_k(results: list[list[str]], relevant: list[set[str]], k: int) -> float:
    """Recall@K: fraction of relevant items in top-K."""
    total = 0.0
    for res, rel in zip(results, relevant):
        top_k = set(res[:k])
        if rel:
            hit = len(top_k & rel) / len(rel)
            total += min(1.0, hit)
        else:
            total += 0.0
    return total / len(results) if results else 0.0


def compute_ap(results: list[str], relevant: set[str]) -> float:
    """Average Precision for a single query."""
    if not relevant:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for i, doc in enumerate(results):
        if doc in relevant:
            hits += 1
            prec_sum += hits / (i + 1)
    return prec_sum / len(relevant) if relevant else 0.0


def compute_map(results: list[list[str]], relevant: list[set[str]]) -> float:
    """Mean Average Precision."""
    return sum(compute_ap(r, rel) for r, rel in zip(results, relevant)) / len(results) if results else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (Recall@K, mAP)")
    parser.add_argument("--results-file", type=Path, help="JSON file with query -> top-K ids")
    parser.add_argument("--ground-truth", type=Path, help="JSON file with query -> set of relevant ids")
    args = parser.parse_args()

    if not args.results_file or not args.ground_truth:
        print("Usage: python evaluate.py --results-file results.json --ground-truth gt.json")
        print("Format: each JSON is a list of {query_id, result_ids} and {query_id, relevant_ids}")
        sys.exit(1)

    import json
    with open(args.results_file) as f:
        results_data = json.load(f)
    with open(args.ground_truth) as f:
        gt_data = json.load(f)

    # Build lookup
    gt_map = {r["query_id"]: set(r["relevant_ids"]) for r in gt_data}
    results_list = []
    relevant_list = []
    for r in results_data:
        qid = r["query_id"]
        if qid in gt_map:
            results_list.append(r["result_ids"])
            relevant_list.append(gt_map[qid])

    r1 = compute_recall_at_k(results_list, relevant_list, 1)
    r5 = compute_recall_at_k(results_list, relevant_list, 5)
    r10 = compute_recall_at_k(results_list, relevant_list, 10)
    mAP = compute_map(results_list, relevant_list)

    print(f"Recall@1:  {r1:.4f}")
    print(f"Recall@5:  {r5:.4f}")
    print(f"Recall@10: {r10:.4f}")
    print(f"mAP:       {mAP:.4f}")


if __name__ == "__main__":
    main()
