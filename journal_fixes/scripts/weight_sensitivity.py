"""
weight_sensitivity.py — does the conclusion depend on the hand-set Global Score weights?

For every matched comparison stored in the repo (same queries, same judge),
reports the paired difference under
  * the paper's weights (70% on Grounding + Specificity),
  * equal weights,
  * each single dimension,
  * 10,000 random weight vectors drawn uniformly from the simplex (Dirichlet(1)),
with bootstrap 95% CIs and Wilcoxon signed-rank p-values. No LLM calls.

Writes journal_fixes/results/weight_sensitivity.json

Usage
  python journal_fixes/scripts/weight_sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "journal_fixes" / "results"
DIMS = ["Grounding", "Specificity", "Correctness", "Completeness", "Code_Quality", "Coherence"]
PAPER_W = np.array([0.35, 0.35, 0.15, 0.10, 0.03, 0.02])
rng = np.random.default_rng(0)


def mode_scores(fname: str) -> dict[str, np.ndarray]:
    d = json.loads((ROOT / "results" / fname).read_text())
    out = {}
    for r in d["runs"]:
        j = r.get("judge") or {}
        if all(k.lower() in j for k in DIMS):
            out[r["query"]] = np.array([j[k.lower()] for k in DIMS], dtype=float)
    return out


def comparison_scores(fname: str, system: str) -> dict[str, np.ndarray]:
    d = json.loads((ROOT / fname).read_text())
    return {r["query"]: np.array([r[system]["scores"][k] for k in DIMS], dtype=float) for r in d}


def paired(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
    qs = sorted(set(a) & set(b))
    return np.stack([a[q] for q in qs]), np.stack([b[q] for q in qs])


def diff_stats(da: np.ndarray, db: np.ndarray, w: np.ndarray) -> dict:
    x, y = da @ w, db @ w
    d = x - y
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    p = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
    return {"a": round(float(x.mean()), 3), "b": round(float(y.mean()), 3), "diff": round(float(d.mean()), 3),
            "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
            "wilcoxon_p": round(float(p), 4)}


def analyse(name: str, A: np.ndarray, B: np.ndarray) -> dict:
    res = {"n": int(len(A)), "paper_weights": diff_stats(A, B, PAPER_W),
           "equal_weights": diff_stats(A, B, np.ones(6) / 6)}
    for i, k in enumerate(DIMS):
        e = np.zeros(6); e[i] = 1
        res[k] = diff_stats(A, B, e)
    W = rng.dirichlet(np.ones(6), size=10000)
    wins = ((A.mean(0) - B.mean(0)) @ W.T) > 0
    res["random_weights_share_a_wins"] = round(float(wins.mean()), 3)
    return res


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    react_ctx = mode_scores("eval_react_cot_context.json")
    react_no = mode_scores("eval_react_cot_no_context.json")
    adaptive = mode_scores("eval_adaptive.json")
    comps = {
        "SciWeaver (full PoT, all 30) vs ReAct with context": paired(adaptive, react_ctx),
        "SciWeaver (full PoT, all 30) vs ReAct no context": paired(adaptive, react_no),
        "ReAct with context vs ReAct no context": paired(react_ctx, react_no),
    }
    for f in ["results/llm_baseline_comparison/comparison_results.json", "results/full_comparison/comparison_results.json"]:
        ad = comparison_scores(f, "adaptive")
        for base in ["gemini", "gpt"]:
            comps[f"SciWeaver vs {base} (50 queries, {f.split('/')[1]})"] = paired(ad, comparison_scores(f, base))

    report = {name: analyse(name, *ab) for name, ab in comps.items()}
    (OUT / "weight_sensitivity.json").write_text(json.dumps(report, indent=1))
    for name, r in report.items():
        pw, ew = r["paper_weights"], r["equal_weights"]
        print(f"{name}\n   n={r['n']}  paper-weights diff {pw['diff']:+.2f} {pw['ci95']} p={pw['wilcoxon_p']}"
              f" | equal-weights diff {ew['diff']:+.2f} {ew['ci95']} p={ew['wilcoxon_p']}"
              f" | Correctness {r['Correctness']['diff']:+.2f} p={r['Correctness']['wilcoxon_p']}"
              f" | A wins under {100 * r['random_weights_share_a_wins']:.0f}% of random weightings")


if __name__ == "__main__":
    main()
