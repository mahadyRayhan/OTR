"""
Statistical rigor pass for OTR evaluation results.

Tests:
  1. Full comparison (N=50): adaptive vs gemini vs gpt — GS_Score + per-dimension
  2. Best-30 comparison: same restricted to best_questions.json query IDs
  3. Profiler ablation (N=30): with vs without profiler — GS_Score + per-dimension
     + binomial test on 22/30 win count

Methods:
  - Bootstrap 95% CI (10 000 resamples, percentile method)
  - Wilcoxon signed-rank test (two-sided, paired)
  - Binomial exact test (for win counts)

Outputs:
  results/stats_rigor/
    stats_rigor_report.txt
    stats_rigor_data.json
    stats_bootstrap_chart.png   (CI forest plot)
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
from scipy import stats

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
OUT_DIR = ROOT / "results" / "stats_rigor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CR_PATH  = ROOT / "results" / "full_comparison" / "comparison_results.json"
PA_PATH  = ROOT / "results" / "profiler_ablation" / "profiler_ablation_results.json"
BQ_PATH  = ROOT / "best_questions.json"

DIMENSIONS = ["Grounding", "Specificity", "Correctness", "Completeness",
              "Code_Quality", "Coherence"]

N_BOOT  = 10_000
ALPHA   = 0.05
RNG     = np.random.default_rng(42)

# ── bootstrap CI ─────────────────────────────────────────────────────────────

def boot_ci(diffs: np.ndarray, n: int = N_BOOT, alpha: float = ALPHA
            ) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using percentile bootstrap on paired differences."""
    means = np.array([
        RNG.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n)
    ])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), float(lo), float(hi)


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float, str]:
    """Paired Wilcoxon signed-rank test. Returns (statistic, p, stars)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    p = float(res.pvalue)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·" if p < 0.1 else "ns"
    return float(res.statistic), p, stars


def binomial_test(k: int, n: int, p0: float = 0.5) -> tuple[float, str]:
    """Exact binomial test (two-sided) for H0: P(win) = p0."""
    res = stats.binomtest(k, n, p0, alternative="two-sided")
    p   = float(res.pvalue)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "·" if p < 0.1 else "ns"
    return p, stars


def pct_change(a: float, b: float) -> str:
    """Return '+X.X%' string for (a-b)/b × 100."""
    if b == 0:
        return "N/A"
    v = (a - b) / b * 100
    return f"{v:+.1f}%"

# ── data loading ─────────────────────────────────────────────────────────────

def load_comparison() -> list[dict]:
    return json.loads(CR_PATH.read_text())


def load_profiler() -> list[dict]:
    return json.loads(PA_PATH.read_text())


def best_qids() -> set[str]:
    return {e["qid"] for e in json.loads(BQ_PATH.read_text())}


def extract_scores(records: list[dict], system: str, dim: str) -> np.ndarray:
    return np.array([r[system]["scores"][dim] for r in records], dtype=float)

# ── analysis helpers ──────────────────────────────────────────────────────────

def compare_systems(records: list[dict], sys_a: str, sys_b: str,
                    label: str) -> list[dict]:
    """Return one result dict per metric (gs_score + each dimension)."""
    results = []
    for dim in ["gs_score"] + DIMENSIONS:
        a = extract_scores(records, sys_a, dim)
        b = extract_scores(records, sys_b, dim)
        d = a - b
        mean, lo, hi = boot_ci(d)
        stat, p, stars = wilcoxon(a, b)
        results.append({
            "comparison": label,
            "metric": dim,
            "n": len(records),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_diff": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "pct_change": pct_change(a.mean(), b.mean()),
            "wilcoxon_stat": stat,
            "p_value": p,
            "sig": stars,
        })
    return results


def profiler_analysis(records: list[dict]) -> list[dict]:
    """Compare with_profiler vs without_profiler."""
    results = []
    for dim in ["gs_score"] + DIMENSIONS:
        wp = np.array([r["with_profiler"]["scores"][dim] for r in records], dtype=float)
        wo = np.array([r["without_profiler"]["scores"][dim] for r in records], dtype=float)
        d  = wp - wo
        mean, lo, hi = boot_ci(d)
        stat, p, stars = wilcoxon(wp, wo)
        results.append({
            "comparison": "profiler: with vs without",
            "metric": dim,
            "n": len(records),
            "mean_with": float(wp.mean()),
            "mean_without": float(wo.mean()),
            "mean_diff": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "pct_change": pct_change(wp.mean(), wo.mean()),
            "wilcoxon_stat": stat,
            "p_value": p,
            "sig": stars,
        })
    # win-count binomial test
    wins = sum(1 for r in records if r["gs_delta"] > 0)
    ties = sum(1 for r in records if r["gs_delta"] == 0)
    losses = sum(1 for r in records if r["gs_delta"] < 0)
    p_bin, stars_bin = binomial_test(wins, len(records))
    results.append({
        "comparison": "profiler win count",
        "metric": "gs_score wins",
        "n": len(records),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_binomial": p_bin,
        "sig": stars_bin,
    })
    return results

# ── report formatting ─────────────────────────────────────────────────────────

def fmt_row(label: str, n: int, mean_a: float, mean_b: float,
            diff: float, lo: float, hi: float, pct: str,
            p: float, sig: str) -> str:
    return (f"  {label:<28} n={n:<4} "
            f"A={mean_a:4.2f}  B={mean_b:4.2f}  "
            f"Δ={diff:+5.3f}  95%CI[{lo:+5.3f},{hi:+5.3f}]  "
            f"{pct:>7}  p={p:.4f}{sig:>4}")


def write_report(all_results: dict, path: pathlib.Path) -> None:
    lines = [
        "STATISTICAL RIGOR REPORT",
        "=" * 78,
        "",
        "  Bootstrap 95% CI: percentile method, N=10 000 resamples",
        "  Significance: Wilcoxon signed-rank (paired, two-sided)",
        "  *** p<0.001  ** p<0.01  * p<0.05  · p<0.1  ns = not significant",
        "",
    ]

    for section_name, rows in all_results.items():
        lines += [f"── {section_name} {'─'*(74-len(section_name))}", ""]
        for row in rows:
            if "wins" in row:
                # binomial row
                lines.append(
                    f"  Win/Tie/Loss: {row['wins']}/{row['ties']}/{row['losses']}  "
                    f"(N={row['n']})  binomial p={row['p_binomial']:.4f}{row['sig']:>4}"
                )
            else:
                key_a = "mean_a" if "mean_a" in row else "mean_with"
                key_b = "mean_b" if "mean_b" in row else "mean_without"
                lines.append(fmt_row(
                    row["metric"], row["n"],
                    row[key_a], row[key_b],
                    row["mean_diff"], row["ci_lo"], row["ci_hi"],
                    row["pct_change"], row["p_value"], row["sig"]
                ))
        lines.append("")

    lines += ["=" * 78]
    path.write_text("\n".join(lines))


# ── forest plot ───────────────────────────────────────────────────────────────

def make_forest_plot(all_results: dict, path: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping forest plot")
        return

    # Collect gs_score rows only
    plot_rows: list[tuple[str, float, float, float, str]] = []
    for section_name, rows in all_results.items():
        for row in rows:
            if row.get("metric") == "gs_score" and "mean_diff" in row:
                label = section_name.replace("Full comparison: ", "All-50: ")\
                                    .replace("Best-30: ", "Best-30: ")\
                                    .replace("Profiler ablation: ", "Profiler: ")
                plot_rows.append((label, row["mean_diff"], row["ci_lo"],
                                  row["ci_hi"], row["sig"]))

    if not plot_rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_rows) * 0.7 + 1.5)))
    y_pos = list(range(len(plot_rows)))

    colors = []
    for _, diff, lo, hi, sig in plot_rows:
        if sig in ("***", "**", "*"):
            colors.append("#e74c3c" if diff < 0 else "#27ae60")
        elif sig == "·":
            colors.append("#f39c12")
        else:
            colors.append("#95a5a6")

    for i, (label, diff, lo, hi, sig) in enumerate(plot_rows):
        ax.plot([lo, hi], [i, i], color=colors[i], lw=2, zorder=2)
        ax.scatter([diff], [i], color=colors[i], s=60, zorder=3)
        ax.text(max(abs(lo), abs(hi), abs(diff)) * 1.05 + 0.02,
                i, f"Δ={diff:+.3f} {sig}", va="center", fontsize=8)

    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in plot_rows], fontsize=9)
    ax.set_xlabel("Mean GS_Score difference (A − B)")
    ax.set_title("Bootstrap 95% CIs — GS_Score pairwise comparisons")
    ax.invert_yaxis()

    sig_patch   = mpatches.Patch(color="#27ae60", label="sig. positive")
    neg_patch   = mpatches.Patch(color="#e74c3c", label="sig. negative")
    trend_patch = mpatches.Patch(color="#f39c12", label="trend (p<0.1)")
    ns_patch    = mpatches.Patch(color="#95a5a6", label="not significant")
    ax.legend(handles=[sig_patch, neg_patch, trend_patch, ns_patch],
              loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Forest plot → {path}")


# ── dimension heatmap ─────────────────────────────────────────────────────────

def make_dimension_heatmap(all_results: dict, path: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    sections = [s for s in all_results if "win count" not in s]
    metrics  = ["gs_score"] + DIMENSIONS

    # Build matrix: rows=sections × metrics, value = mean_diff
    data = np.zeros((len(sections), len(metrics)))
    sig_mat = [[""] * len(metrics) for _ in range(len(sections))]

    for si, sec in enumerate(sections):
        for row in all_results[sec]:
            if row.get("metric") in metrics and "mean_diff" in row:
                mi = metrics.index(row["metric"])
                data[si, mi] = row["mean_diff"]
                sig_mat[si][mi] = row["sig"]

    vmax = max(abs(data).max(), 0.3)
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.3),
                                    max(4, len(sections) * 0.9 + 1.5)))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(sections)))
    ax.set_yticklabels([s[:45] for s in sections], fontsize=8)

    for si in range(len(sections)):
        for mi in range(len(metrics)):
            v   = data[si, mi]
            ann = f"{v:+.2f}\n{sig_mat[si][mi]}"
            ax.text(mi, si, ann, ha="center", va="center",
                    fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Mean difference (A − B)")
    ax.set_title("Per-dimension mean differences with significance\n"
                 "Green = A better · Red = A worse")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dimension heatmap → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("⏳ Loading data...")
    cr_all  = load_comparison()               # 50 queries
    bqids   = best_qids()
    cr_best = [r for r in cr_all if r["query_id"] in bqids]   # 30 queries
    pa      = load_profiler()                 # 30 queries

    print(f"  Full comparison : N={len(cr_all)}")
    print(f"  Best-30 subset  : N={len(cr_best)}")
    print(f"  Profiler ablation: N={len(pa)}")

    print("\n⏳ Computing statistics...")

    all_results: dict[str, list[dict]] = {}

    # ── 1. Full comparison (N=50) ─────────────────────────────────────────
    for sys_b in ("gemini", "gpt"):
        key = f"Full comparison: adaptive vs {sys_b} (N=50)"
        all_results[key] = compare_systems(cr_all, "adaptive", sys_b, key)

    # adaptive vs best baseline per query
    best_base_scores = {
        "gs_score": np.array([max(r["gemini"]["scores"]["gs_score"],
                                  r["gpt"]["scores"]["gs_score"])
                              for r in cr_all]),
    }
    for dim in DIMENSIONS:
        best_base_scores[dim] = np.array([
            max(r["gemini"]["scores"][dim], r["gpt"]["scores"][dim])
            for r in cr_all
        ])
    adap_scores_all: dict[str, np.ndarray] = {
        "gs_score": extract_scores(cr_all, "adaptive", "gs_score"),
    }
    for dim in DIMENSIONS:
        adap_scores_all[dim] = extract_scores(cr_all, "adaptive", dim)

    rows_vs_best_all = []
    for dim in ["gs_score"] + DIMENSIONS:
        a = adap_scores_all[dim]
        b = best_base_scores[dim]
        d = a - b
        mean, lo, hi = boot_ci(d)
        stat, p, stars = wilcoxon(a, b)
        rows_vs_best_all.append({
            "comparison": "adaptive vs best-baseline (N=50)",
            "metric": dim,
            "n": len(cr_all),
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_diff": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "pct_change": pct_change(a.mean(), b.mean()),
            "wilcoxon_stat": stat,
            "p_value": p,
            "sig": stars,
        })
    all_results["Full comparison: adaptive vs best-baseline (N=50)"] = rows_vs_best_all

    # ── 2. Best-30 subset ─────────────────────────────────────────────────
    for sys_b in ("gemini", "gpt"):
        key = f"Best-30: adaptive vs {sys_b} (N=30)"
        all_results[key] = compare_systems(cr_best, "adaptive", sys_b, key)

    best_base_30 = {
        "gs_score": np.array([max(r["gemini"]["scores"]["gs_score"],
                                  r["gpt"]["scores"]["gs_score"])
                              for r in cr_best]),
    }
    for dim in DIMENSIONS:
        best_base_30[dim] = np.array([
            max(r["gemini"]["scores"][dim], r["gpt"]["scores"][dim])
            for r in cr_best
        ])
    adap_scores_30: dict[str, np.ndarray] = {
        "gs_score": extract_scores(cr_best, "adaptive", "gs_score"),
    }
    for dim in DIMENSIONS:
        adap_scores_30[dim] = extract_scores(cr_best, "adaptive", dim)

    rows_vs_best_30 = []
    for dim in ["gs_score"] + DIMENSIONS:
        a = adap_scores_30[dim]
        b = best_base_30[dim]
        d = a - b
        mean, lo, hi = boot_ci(d)
        stat, p, stars = wilcoxon(a, b)
        rows_vs_best_30.append({
            "comparison": "adaptive vs best-baseline (N=30)",
            "metric": dim,
            "n": 30,
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "mean_diff": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "pct_change": pct_change(a.mean(), b.mean()),
            "wilcoxon_stat": stat,
            "p_value": p,
            "sig": stars,
        })
    all_results["Best-30: adaptive vs best-baseline (N=30)"] = rows_vs_best_30

    # ── 3. Profiler ablation (N=30) ───────────────────────────────────────
    all_results["Profiler ablation (N=30)"] = profiler_analysis(pa)

    # ── print summary to console ──────────────────────────────────────────
    print("\n  KEY HEADLINE RESULTS")
    print("  " + "-" * 74)
    headline_sections = [
        "Full comparison: adaptive vs best-baseline (N=50)",
        "Best-30: adaptive vs best-baseline (N=30)",
        "Profiler ablation (N=30)",
    ]
    for sec in headline_sections:
        print(f"\n  [{sec}]")
        for row in all_results[sec]:
            if row.get("metric") in ("gs_score", "gs_score wins"):
                if "wins" in row:
                    print(f"    Win/Tie/Loss: {row['wins']}/{row['ties']}/{row['losses']}"
                          f"  p={row['p_binomial']:.4f}{row['sig']:>4}")
                else:
                    key_a = "mean_a" if "mean_a" in row else "mean_with"
                    key_b = "mean_b" if "mean_b" in row else "mean_without"
                    print(f"    gs_score: A={row[key_a]:.3f}  B={row[key_b]:.3f}"
                          f"  Δ={row['mean_diff']:+.3f}"
                          f"  95%CI[{row['ci_lo']:+.3f},{row['ci_hi']:+.3f}]"
                          f"  {row['pct_change']}"
                          f"  p={row['p_value']:.4f}{row['sig']:>4}")

    # ── save outputs ──────────────────────────────────────────────────────
    data_path   = OUT_DIR / "stats_rigor_data.json"
    report_path = OUT_DIR / "stats_rigor_report.txt"
    forest_path = OUT_DIR / "stats_bootstrap_chart.png"
    heat_path   = OUT_DIR / "stats_dimension_heatmap.png"

    data_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n  Data   → {data_path}")

    write_report(all_results, report_path)
    print(f"  Report → {report_path}")

    print("\n⏳ Generating charts...")
    make_forest_plot(all_results, forest_path)
    make_dimension_heatmap(all_results, heat_path)

    print(f"\nDone. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
