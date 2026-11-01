"""
eval_baseline_gs.py — Compute / verify GS_Score for GPT and Gemini baselines
==============================================================================
Reads the saved baseline comparison JSON, (re)computes GS_Score from the six
stored dimensions using the canonical formula, and prints a per-question
comparison table plus per-model summary.

No API calls needed — all scores are already in the file.

Data files (newest / most complete first):
    results/llm_baseline_comparison/comparison_results.json        ← 50 entries
    results-best/llm_baseline_comparison/comparison_results.json   ← 35 entries

Usage:
    python scripts/eval/eval_baseline_gs.py
    python scripts/eval/eval_baseline_gs.py --file results-best/llm_baseline_comparison/comparison_results.json
    python scripts/eval/eval_baseline_gs.py --out results/eval_baseline_gs.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# GS_Score weights — must match the rest of the system
# ---------------------------------------------------------------------------
GS_WEIGHTS = {
    "grounding":    0.35,
    "specificity":  0.35,
    "correctness":  0.15,
    "completeness": 0.10,
    "code_quality": 0.03,
    "coherence":    0.02,
}

# The stored keys use Title_Case — map to lowercase weight keys
_KEY_MAP = {
    "Grounding":    "grounding",
    "Specificity":  "specificity",
    "Correctness":  "correctness",
    "Completeness": "completeness",
    "Code_Quality": "code_quality",
    "Coherence":    "coherence",
}

MODELS = ["adaptive", "gemini", "gpt"]

DEFAULT_FILE = ROOT / "results" / "llm_baseline_comparison" / "comparison_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_gs(scores: dict) -> float:
    """Compute GS_Score from a scores dict (Title_Case or lowercase keys)."""
    total = 0.0
    for stored_key, weight_key in _KEY_MAP.items():
        # Accept both Title_Case (stored) and lowercase
        val = scores.get(stored_key, scores.get(weight_key, 0))
        total += GS_WEIGHTS[weight_key] * float(val)
    return round(total, 3)


def mean(lst: list) -> float:
    return round(sum(lst) / len(lst), 3) if lst else 0.0


def load_entries(path: Path) -> list[dict]:
    entries = json.loads(path.read_text())
    # Only keep entries that have all three models with score data
    complete = [
        e for e in entries
        if all(
            m in e and isinstance(e[m].get("scores"), dict)
            for m in MODELS
        )
    ]
    return complete


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n{'═'*105}")
    print(f"  LLM BASELINE COMPARISON — GS_Score  |  {n} questions")
    print(f"{'═'*105}")

    hdr = (
        f"  {'QID':<7} "
        f"{'Adapt_GS':>9} {'Adapt_Grnd':>10} {'Adapt_Spec':>10} "
        f"{'GPT_GS':>8} {'GPT_Grnd':>9} {'GPT_Spec':>9} "
        f"{'Gem_GS':>8} {'Gem_Grnd':>9} {'Gem_Spec':>9}"
    )
    print(hdr)
    print(f"  {'─'*7} {'─'*9} {'─'*10} {'─'*10} {'─'*8} {'─'*9} {'─'*9} {'─'*8} {'─'*9} {'─'*9}")

    for r in rows:
        qid  = r["query_id"]
        a, g, gem = r["adaptive"], r["gpt"], r["gemini"]
        print(
            f"  {qid:<7} "
            f"{a['gs']:>9.3f} {a['scores'].get('Grounding',0):>10.1f} {a['scores'].get('Specificity',0):>10.1f} "
            f"{g['gs']:>8.3f} {g['scores'].get('Grounding',0):>9.1f} {g['scores'].get('Specificity',0):>9.1f} "
            f"{gem['gs']:>8.3f} {gem['scores'].get('Grounding',0):>9.1f} {gem['scores'].get('Specificity',0):>9.1f}"
        )

    # Per-model summary
    print(f"\n{'─'*105}")
    print(f"  SUMMARY (n={n})")
    print(f"\n  {'Metric':<30} {'Adaptive':>10} {'GPT-4':>10} {'Gemini':>10}  {'Best':>8}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}  {'─'*8}")

    dims = list(_KEY_MAP.keys())
    metrics = [("GS_Score", "gs")] + [(d, d) for d in dims]

    for label, key in metrics:
        vals = {}
        for m in MODELS:
            if key == "gs":
                vals[m] = mean([r[m]["gs"] for r in rows])
            else:
                vals[m] = mean([r[m]["scores"].get(key, 0) for r in rows])
        best = max(vals, key=lambda x: vals[x])
        print(
            f"  {label:<30} {vals['adaptive']:>10.3f} {vals['gpt']:>10.3f} {vals['gemini']:>10.3f}  {best:>8}"
        )

    print(f"{'═'*105}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(file_path: Path, out_path: Path | None) -> None:
    print(f"\n── Baseline GS_Score Analysis ────────────────────────────────")
    print(f"  Source: {file_path}")

    if not file_path.exists():
        print(f"  File not found: {file_path}")
        sys.exit(1)

    entries = load_entries(file_path)
    print(f"  Complete entries (all 3 models): {len(entries)}")

    # (Re)compute GS for every model in every entry
    rows = []
    for e in entries:
        row = {"query_id": e["query_id"], "query": e["query"]}
        for m in MODELS:
            s = e[m]["scores"]
            gs = compute_gs(s)
            row[m] = {
                "gs":      gs,
                "scores":  s,
                "answer":  e[m].get("answer", ""),
            }
        rows.append(row)

    rows.sort(key=lambda x: x["query_id"])

    print_table(rows)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Build a clean JSON summary
        summary = {}
        for m in MODELS:
            gs_vals  = [r[m]["gs"] for r in rows]
            dim_vals = {k: mean([r[m]["scores"].get(k, 0) for r in rows]) for k in _KEY_MAP}
            summary[m] = {
                "mean_gs_score": mean(gs_vals),
                "dim_means": dim_vals,
                "per_question": [
                    {"query_id": r["query_id"], "gs": r[m]["gs"], "scores": r[m]["scores"]}
                    for r in rows
                ],
            }
        out = {
            "source_file": str(file_path),
            "n_questions": len(rows),
            "gs_weights":  GS_WEIGHTS,
            "models":      summary,
        }
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  → Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None,
                        help="Input comparison JSON (default: results/llm_baseline_comparison/comparison_results.json)")
    parser.add_argument("--out",  type=Path, default=None,
                        help="Output JSON path (default: no file saved)")
    args = parser.parse_args()

    src = args.file or DEFAULT_FILE
    main(src, args.out)
