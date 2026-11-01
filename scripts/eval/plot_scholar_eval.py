"""
plot_scholar_eval.py — Scholar Agent evaluation summary table
Usage:
    python scripts/eval/plot_scholar_eval.py
    python scripts/eval/plot_scholar_eval.py --input results/eval_scholar.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PASS_THRESHOLD = 7
GROUPS = {
    "G1 Specific Bio":  ["SQ01", "SQ02", "SQ03"],
    "G2 Multi-omics":   ["SQ04", "SQ05"],
    "G3 Computational": ["SQ06", "SQ07"],
    "G4 Collab":        ["SQ08", "SQ09"],
    "G5 Broad":         ["SQ10"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "eval_scholar.json")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Not found: {args.input}")
        sys.exit(1)

    data    = json.loads(args.input.read_text())
    results = data["results"]

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n  Pass {data['pass_rate_pct']:.0f}%  |  Mean {data['mean_judge_score']:.1f}/10  |  Lat {data['mean_latency_s']:.1f}s\n")

    # ── Group table ───────────────────────────────────────────────────────────
    print(f"  {'Group':<20} {'N':>2}  {'Mean':>5}  {'Pass%':>6}  {'Rel/4':>6}  {'Comp/3':>7}  {'Use/3':>6}")
    print(f"  {'─'*20} {'─'*2}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")
    for grp_name, qids in GROUPS.items():
        rs = [r for r in results if r["query_id"] in qids]
        if not rs:
            continue
        n      = len(rs)
        mean_s = sum(r["judge"]["total"]        for r in rs) / n
        pass_r = sum(r["passed"]                for r in rs) / n * 100
        mean_r = sum(r["judge"]["relevance"]    for r in rs) / n
        mean_c = sum(r["judge"]["completeness"] for r in rs) / n
        mean_u = sum(r["judge"]["usefulness"]   for r in rs) / n
        print(
            f"  {grp_name:<20} {n:>2}  {mean_s:>5.1f}  {pass_r:>5.0f}%  "
            f"{mean_r:>6.1f}  {mean_c:>7.1f}  {mean_u:>6.1f}"
        )

    # ── Overall row ───────────────────────────────────────────────────────────
    n_all  = len(results)
    mean_r = sum(r["judge"]["relevance"]    for r in results) / n_all
    mean_c = sum(r["judge"]["completeness"] for r in results) / n_all
    mean_u = sum(r["judge"]["usefulness"]   for r in results) / n_all
    print(f"  {'─'*20} {'─'*2}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}")
    print(
        f"  {'All':<20} {n_all:>2}  {data['mean_judge_score']:>5.1f}  "
        f"{data['pass_rate_pct']:>5.0f}%  {mean_r:>6.1f}  {mean_c:>7.1f}  {mean_u:>6.1f}"
    )
    print()


if __name__ == "__main__":
    main()
