"""
audit_table1.py — trace every number in the ICDCN Table 1 back to stored results,
and check the evaluation for selection effects. No LLM calls.

Writes journal_fixes/results/audit_table1.json and prints a readable report.

Usage
  python journal_fixes/scripts/audit_table1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "journal_fixes" / "results"
W = {"Grounding": 0.35, "Specificity": 0.35, "Correctness": 0.15, "Completeness": 0.10, "Code_Quality": 0.03, "Coherence": 0.02}
DIMS = list(W)

# Table 1 as printed in the accepted paper (GS, Corr, Comp, Grounding, Spec, Code, Coh)
PAPER = {
    "GPT-5 Baseline":            (4.43, 3.77, 3.57, 1.27, 4.30, 1.80, 6.27),
    "Gemini-3 Baseline":         (3.88, 4.37, 3.73, 1.80, 5.10, 2.80, 6.80),
    "ReAct (No Context)":        (6.05, 4.63, 6.00, 5.97, 6.87, 3.47, 7.97),
    "ReAct (With Context)":      (6.31, 4.83, 6.33, 6.27, 7.10, 3.77, 8.10),
    "SciWeaver (with PoT)":      (6.77, 5.77, 6.57, 6.57, 7.53, 4.77, 8.40),
    "SciWeaver (without PoT)":   (6.31, 5.20, 6.00, 6.03, 7.20, 4.70, 8.00),
}


def gs(row: dict) -> float:
    return sum(W[k] * row[k] for k in DIMS)


def load_queries(path: Path) -> list[str]:
    q = json.loads(path.read_text())
    return q if isinstance(q[0], str) else [x.get("query") for x in q]


def eval_mode(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {"gs": d["summary"]["mean_gs_score"], "dims": d["summary"]["dim_means"],
            "queries": [r["query"] for r in d["runs"]], "n": d["n_runs"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"weighted_formula": W}
    best = set(load_queries(ROOT / "best_questions.json"))

    # 1. Paper rows: is each GS consistent with its own dimension scores?
    consistency = {}
    for name, (g, c, comp, gr, sp, code, coh) in PAPER.items():
        re_gs = gs({"Correctness": c, "Completeness": comp, "Grounding": gr, "Specificity": sp, "Code_Quality": code, "Coherence": coh})
        consistency[name] = {"printed_gs": g, "gs_from_printed_dims": round(re_gs, 2), "consistent": abs(re_gs - g) < 0.02}
    report["paper_row_consistency"] = consistency

    # 2. Where each row comes from
    modes = {name: eval_mode(ROOT / "results" / f) for name, f in [
        ("react_no_context", "eval_react_cot_no_context.json"),
        ("react_context", "eval_react_cot_context.json"),
        ("adaptive_with_pot_all30", "eval_adaptive.json"),
        ("adaptive_best_of_pool", "eval_best_adaptive.json"),
    ]}
    report["stored_runs"] = {k: {"gs": v["gs"], "n": v["n"], "queries_in_benchmark": len(set(v["queries"]) & best)} for k, v in modes.items()}

    # 3. Was the 30-query benchmark selected by outcome?
    fc = json.loads((ROOT / "results/full_comparison/comparison_results.json").read_text())
    rows = [(r["query"], r["adaptive"]["scores"]["gs_score"],
             max(r["gemini"]["scores"]["gs_score"], r["gpt"]["scores"]["gs_score"]),
             r["gemini"]["scores"]["gs_score"], r["gpt"]["scores"]["gs_score"]) for r in fc]
    adv = sorted(rows, key=lambda x: x[1] - x[2], reverse=True)
    top30 = {x[0] for x in adv[:30]}
    inb = [x for x in rows if x[0] in best]
    outb = [x for x in rows if x[0] not in best]

    def summary(S):
        a = np.array([x[1] for x in S]); b = np.array([x[2] for x in S]); g = np.array([x[3] for x in S])
        p = stats.wilcoxon(a, g).pvalue if len(S) > 5 else None
        return {"n": len(S), "sciweaver": round(a.mean(), 3), "best_baseline": round(b.mean(), 3),
                "gemini": round(g.mean(), 3), "diff_vs_best": round((a - b).mean(), 3),
                "wilcoxon_p_vs_gemini": None if p is None else round(float(p), 4)}

    report["selection_check"] = {
        "benchmark_equals_top30_by_sciweaver_advantage": len(top30 & best) == 30,
        "overlap": len(top30 & best),
        "benchmark_30": summary(inb), "excluded_20": summary(outb), "all_50": summary(rows),
    }

    # 4. Matched-run comparison: the one run where all three systems answered the same 50 queries
    for p in ["results/llm_baseline_comparison/comparison_results.json", "results/full_comparison/comparison_results.json"]:
        d = json.loads((ROOT / p).read_text())
        res = {}
        for s in ["adaptive", "gemini", "gpt"]:
            sc = [r[s]["scores"] for r in d]
            dm = {k: float(np.mean([x[k] for x in sc])) for k in DIMS}
            res[s] = {"gs": round(gs(dm), 2), "unweighted": round(float(np.mean(list(dm.values()))), 2), **{k: round(v, 2) for k, v in dm.items()}}
        report.setdefault("matched_runs_50q", {})[p] = res

    (OUT / "audit_table1.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
