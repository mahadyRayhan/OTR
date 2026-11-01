"""
run_cugcr_correlation.py
========================
Measure the statistical relationship between CUGCR compression metrics
and downstream answer quality (both internal judge and GPT GS_Score).

CUGCR features extracted per adaptive run
------------------------------------------
  cugcr_triggered      — bool: did CUGCR fire at least once in this run?
  n_cugcr_nodes        — int: how many DAG nodes had CUGCR active?
  mean_compression_ratio — float: mean(compressed_ctx_tokens / raw_ctx_tokens) across edges
  min_compression_ratio  — float: worst-case compression edge in the run
  mean_crs_score         — float: mean CRS gate cosine-similarity score across edges
  min_crs_score          — float: lowest CRS score seen (weakest upstream relevance)
  n_over_compressed      — int: edges where ratio < 0.15 (risky info loss)
  pct_over_compressed    — float: fraction of edges that are over-compressed

Quality metrics used
---------------------
  internal_score  — final_judge_score in engine_telemetry (0-10 int)
  gs_score        — GPT-as-judge GS_Score from full_comparison/comparison_results.json
                    (weighted: 0.35×Grounding + 0.35×Specificity + ...)
                    Matched by query prefix (120 chars). For queries with multiple
                    adaptive runs, each run is matched to the same gs_score.

Outputs
-------
  results/cugcr_correlation/
    cugcr_correlation_data.json    — per-run feature table
    cugcr_correlation_report.txt   — Pearson r + Spearman ρ table + interpretation
    cugcr_scatter_plots.png        — scatter matrix: each feature vs both quality metrics
    cugcr_correlation_heatmap.png  — feature × quality correlation heatmap

Usage
-----
  python scripts/run_cugcr_correlation.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEL_DIR     = ROOT / "raw_telemetry"
CMP_RESULTS = ROOT / "results" / "full_comparison" / "comparison_results.json"
OUT_DIR     = ROOT / "results" / "cugcr_correlation"

# ── Compression ratio helper (strips task-query overhead) ─────────────────────

def _context_block_tokens(raw_context_injected: str, original_query: str) -> int:
    sep = "---\n## Your Task\n"
    idx = raw_context_injected.rfind(sep)
    if idx >= 0:
        ctx = raw_context_injected[:idx]
    else:
        ctx = raw_context_injected[: max(0, len(raw_context_injected) - len(original_query) - 25)]
    return max(1, len(ctx) // 4)


# ── Build per-run feature table ───────────────────────────────────────────────

def _extract_features(tel_dir: Path) -> list[dict]:
    """Return one record per adaptive run with CUGCR features + internal score."""
    records = []
    for fn in sorted(tel_dir.glob("run_*.json")):
        try:
            data = json.loads(fn.read_text())
        except Exception:
            continue
        if data.get("ablation_mode") != "adaptive":
            continue
        et = data.get("engine_telemetry", {})
        edges = et.get("communication_edges", [])
        nodes = et.get("got_dag_nodes", [])

        if not edges:
            continue

        ratios, crs_list = [], []
        n_oc = 0
        for e in edges:
            ctx_tok = e.get("context_token_count", 0)
            if ctx_tok <= 0:
                continue
            raw_ci = e.get("raw_context_injected", "")
            orig_q = e.get("original_query", "")
            out_tok = _context_block_tokens(raw_ci or e.get("resolved_query", ""), orig_q)
            ratio = out_tok / ctx_tok
            ratios.append(ratio)
            if ratio < 0.15:
                n_oc += 1
            crs = e.get("context_relevance")
            if crs is not None:
                crs_list.append(float(crs))

        if not ratios:
            continue

        n_cugcr_nodes = sum(1 for n in nodes if n.get("cugcr_triggered"))

        records.append({
            "query":                 et.get("user_query", "").strip()[:120],
            "run_id":                data.get("run_id", ""),
            "cugcr_triggered":       int(et.get("cugcr_triggered", False)),
            "n_cugcr_nodes":         n_cugcr_nodes,
            "mean_compression_ratio": round(float(np.mean(ratios)), 4),
            "min_compression_ratio":  round(float(np.min(ratios)), 4),
            "mean_crs_score":         round(float(np.mean(crs_list)), 4) if crs_list else None,
            "min_crs_score":          round(float(np.min(crs_list)), 4)  if crs_list else None,
            "n_over_compressed":      n_oc,
            "pct_over_compressed":    round(n_oc / len(ratios), 4),
            "internal_score":         et.get("final_judge_score"),
        })
    return records


def _attach_gs_scores(records: list[dict], cmp_path: Path) -> list[dict]:
    """Join GPT GS_Score from comparison_results.json by query prefix."""
    if not cmp_path.exists():
        print(f"  ⚠  {cmp_path} not found — skipping GS_Score join")
        return records

    cmp = json.loads(cmp_path.read_text())
    gs_map: dict[str, float] = {}
    for entry in cmp:
        q   = entry.get("query", "").strip()[:120]
        sc  = entry.get("adaptive", {}).get("scores", {}).get("gs_score")
        if q and sc is not None:
            gs_map[q] = float(sc)

    for r in records:
        r["gs_score"] = gs_map.get(r["query"])
    return records


# ── Correlation analysis ──────────────────────────────────────────────────────

FEATURES = [
    ("cugcr_triggered",        "CUGCR triggered (0/1)"),
    ("n_cugcr_nodes",          "# CUGCR nodes"),
    ("mean_compression_ratio", "Mean compression ratio"),
    ("min_compression_ratio",  "Min compression ratio"),
    ("mean_crs_score",         "Mean CRS score"),
    ("min_crs_score",          "Min CRS score"),
    ("n_over_compressed",      "# over-compressed edges"),
    ("pct_over_compressed",    "% over-compressed edges"),
]
QUALITY_METRICS = [
    ("internal_score", "Internal judge (0–10)"),
    ("gs_score",       "GPT GS_Score (0–10)"),
]


def compute_correlations(records: list[dict]) -> dict:
    results = {}
    for feat_key, feat_label in FEATURES:
        results[feat_key] = {}
        for q_key, q_label in QUALITY_METRICS:
            pairs = [
                (r[feat_key], r[q_key])
                for r in records
                if r.get(feat_key) is not None and r.get(q_key) is not None
            ]
            if len(pairs) < 5:
                results[feat_key][q_key] = {"n": len(pairs), "pearson_r": None, "spearman_rho": None}
                continue
            x, y = zip(*pairs)
            x, y = np.array(x, float), np.array(y, float)
            pr, pp = sp_stats.pearsonr(x, y)
            sr, sp = sp_stats.spearmanr(x, y)
            results[feat_key][q_key] = {
                "n":            len(pairs),
                "pearson_r":    round(float(pr), 4),
                "pearson_p":    round(float(pp), 4),
                "spearman_rho": round(float(sr), 4),
                "spearman_p":   round(float(sp), 4),
            }
    return results


def _interpret(r: float | None) -> str:
    if r is None: return "n/a"
    a = abs(r)
    sign = "+" if r > 0 else "−"
    if a >= 0.5:  return f"{sign}strong"
    if a >= 0.3:  return f"{sign}moderate"
    if a >= 0.1:  return f"{sign}weak"
    return "none"


# ── Charts ────────────────────────────────────────────────────────────────────

def _make_scatter_plots(records: list[dict], out_path: Path):
    """One scatter plot per (feature × quality_metric) combination."""
    n_feat = len(FEATURES)
    n_qual = len(QUALITY_METRICS)
    fig, axes = plt.subplots(n_feat, n_qual,
                             figsize=(6 * n_qual, 3.2 * n_feat),
                             constrained_layout=True)
    fig.patch.set_facecolor("white")

    for fi, (feat_key, feat_label) in enumerate(FEATURES):
        for qi, (q_key, q_label) in enumerate(QUALITY_METRICS):
            ax = axes[fi][qi]
            pairs = [
                (r[feat_key], r[q_key])
                for r in records
                if r.get(feat_key) is not None and r.get(q_key) is not None
            ]
            if len(pairs) < 3:
                ax.set_visible(False)
                continue
            x, y = zip(*pairs)
            x, y = np.array(x, float), np.array(y, float)

            # Scatter
            ax.scatter(x, y, alpha=0.45, s=20, color="#3498db", edgecolors="none")

            # Regression line
            if len(x) >= 3 and np.std(x) > 0:
                m, b, r_val, p_val, _ = sp_stats.linregress(x, y)
                xr = np.linspace(x.min(), x.max(), 100)
                ax.plot(xr, m * xr + b, color="#e74c3c", linewidth=1.5)
                sr, sp = sp_stats.spearmanr(x, y)
                sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
                ax.set_title(
                    f"r={r_val:.3f}{sig}  ρ={sr:.3f}  n={len(x)}",
                    fontsize=8, fontweight="bold"
                )
            else:
                ax.set_title(f"n={len(x)}", fontsize=8)

            ax.set_xlabel(feat_label, fontsize=7)
            ax.set_ylabel(q_label,    fontsize=7)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("CUGCR Features vs Answer Quality  —  Scatter + Regression",
                 fontsize=12, fontweight="bold", y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plots → {out_path}")


def _make_heatmap(corr: dict, out_path: Path):
    """Heatmap of Pearson r and Spearman ρ side-by-side."""
    feat_labels = [lbl for _, lbl in FEATURES]
    qual_labels = [lbl for _, lbl in QUALITY_METRICS]

    pearson  = np.full((len(FEATURES), len(QUALITY_METRICS)), np.nan)
    spearman = np.full((len(FEATURES), len(QUALITY_METRICS)), np.nan)
    for fi, (fk, _) in enumerate(FEATURES):
        for qi, (qk, _) in enumerate(QUALITY_METRICS):
            v = corr.get(fk, {}).get(qk, {})
            pearson[fi, qi]  = v.get("pearson_r")  if v.get("pearson_r")  is not None else np.nan
            spearman[fi, qi] = v.get("spearman_rho") if v.get("spearman_rho") is not None else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    fig.patch.set_facecolor("white")

    for ax, mat, title in zip(axes, [pearson, spearman], ["Pearson r", "Spearman ρ"]):
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(qual_labels)))
        ax.set_xticklabels(qual_labels, fontsize=9, rotation=15, ha="right")
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
        for fi in range(len(FEATURES)):
            for qi in range(len(QUALITY_METRICS)):
                v = mat[fi, qi]
                if not np.isnan(v):
                    ax.text(qi, fi, f"{v:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if abs(v) > 0.5 else "black")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    fig.suptitle("CUGCR Feature × Quality Correlation Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap → {out_path}")


# ── Text report ───────────────────────────────────────────────────────────────

def _write_report(records: list, corr: dict, out_path: Path):
    n_runs = len(records)
    n_with_gs = sum(1 for r in records if r.get("gs_score") is not None)
    n_cugcr   = sum(1 for r in records if r.get("cugcr_triggered"))

    lines = [
        "CUGCR ↔ QUALITY CORRELATION REPORT",
        "=" * 70,
        "",
        f"  Adaptive runs analyzed : {n_runs}",
        f"  Runs with GS_Score     : {n_with_gs}",
        f"  Runs where CUGCR fired : {n_cugcr} ({100*n_cugcr/n_runs:.0f}%)",
        "",
        "  Significance codes: *** p<0.001  ** p<0.01  * p<0.05  · p<0.1",
        "",
        f"  {'Feature':<30}  {'Quality metric':<26}  {'r':>6}  {'ρ':>6}  {'n':>4}  Interpretation",
        f"  {'-'*90}",
    ]

    for feat_key, feat_label in FEATURES:
        for q_key, q_label in QUALITY_METRICS:
            v = corr.get(feat_key, {}).get(q_key, {})
            pr  = v.get("pearson_r")
            sr  = v.get("spearman_rho")
            pp  = v.get("pearson_p", 1.0)
            sp  = v.get("spearman_p", 1.0)
            n   = v.get("n", 0)
            def sig(p): return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ("·" if p < 0.1 else " ")))
            pr_s = f"{pr:+.3f}{sig(pp)}" if pr is not None else "   n/a"
            sr_s = f"{sr:+.3f}{sig(sp)}" if sr is not None else "   n/a"
            interp = _interpret(pr)
            lines.append(
                f"  {feat_label:<30}  {q_label:<26}  {pr_s:>7}  {sr_s:>7}  {n:>4}  {interp}"
            )
        lines.append("")

    # Summary: top correlations
    ranked = []
    for fk, fl in FEATURES:
        for qk, ql in QUALITY_METRICS:
            v = corr.get(fk, {}).get(qk, {})
            if v.get("pearson_r") is not None:
                ranked.append((abs(v["pearson_r"]), v["pearson_r"], fl, ql, v.get("pearson_p", 1)))
    ranked.sort(reverse=True)

    lines += [
        "  TOP CORRELATIONS (by |Pearson r|)",
        f"  {'-'*60}",
    ]
    for i, (absr, r, fl, ql, p) in enumerate(ranked[:6], 1):
        def sig2(p): return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        lines.append(f"  {i}. r={r:+.3f}{sig2(p)}  {fl}  →  {ql}")

    lines += [
        "",
        "  INTERPRETATION GUIDE",
        "  |r| ≥ 0.5  strong   |r| 0.3-0.5  moderate",
        "  |r| 0.1-0.3  weak   |r| < 0.1  none",
        "  Negative r → higher feature value = lower quality (bad)",
        "  Positive r → higher feature value = higher quality (good)",
        "",
        "=" * 70,
    ]
    out_path.write_text("\n".join(lines))
    print(f"  Report → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("⏳ Extracting CUGCR features from telemetry...")
    records = _extract_features(TEL_DIR)
    records = _attach_gs_scores(records, CMP_RESULTS)
    print(f"  Runs: {len(records)}  |  with GS_Score: {sum(1 for r in records if r.get('gs_score') is not None)}")

    data_path = OUT_DIR / "cugcr_correlation_data.json"
    data_path.write_text(json.dumps(records, indent=2))
    print(f"  Data → {data_path}")

    print("\n⏳ Computing correlations...")
    corr = compute_correlations(records)

    # Quick console preview
    print(f"\n  {'Feature':<30}  {'vs Internal':>12}  {'vs GS_Score':>12}")
    print(f"  {'-'*58}")
    for fk, fl in FEATURES:
        vi = corr[fk].get("internal_score", {})
        vg = corr[fk].get("gs_score", {})
        ri = f"{vi['pearson_r']:+.3f}" if vi.get("pearson_r") is not None else "  n/a"
        rg = f"{vg['pearson_r']:+.3f}" if vg.get("pearson_r") is not None else "  n/a"
        print(f"  {fl:<30}  {ri:>12}  {rg:>12}")

    print("\n⏳ Generating charts...")
    _make_scatter_plots(records, OUT_DIR / "cugcr_scatter_plots.png")
    _make_heatmap(corr, OUT_DIR / "cugcr_correlation_heatmap.png")
    _write_report(records, corr, OUT_DIR / "cugcr_correlation_report.txt")

    print("\nDone. All outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
