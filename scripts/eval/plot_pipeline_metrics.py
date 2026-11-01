"""
plot_pipeline_metrics.py — Latency / Success Rate / Token Utilization by ablation mode
Usage:
    python scripts/eval/plot_pipeline_metrics.py
    python scripts/eval/plot_pipeline_metrics.py --dir historic_telemetry/raw_telemetry_best
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MODES  = ["none", "raw", "adaptive"]
COLORS = {"none": "#3498db", "raw": "#e67e22", "adaptive": "#2ecc71"}


def load_runs(tel_dir: Path) -> dict[str, list]:
    by_mode = defaultdict(list)
    for f in sorted(tel_dir.glob("run_*.json")):
        try:
            d    = json.loads(f.read_text())
            mode = d.get("ablation_mode", "")
            if mode in MODES:
                by_mode[mode].append(d)
        except Exception:
            pass
    return by_mode


def compute_metrics(runs: list) -> dict:
    latencies, fps_vals, tokens = [], [], []
    for r in runs:
        et = r.get("engine_telemetry", {})

        # latency — prefer et field, fall back to wall_latency_ms
        lat = et.get("latency_seconds") or r.get("wall_latency_ms", 0) / 1000
        if lat:
            latencies.append(lat)

        # first-pass success — prefer boolean field, fall back to retries==0
        fps = et.get("first_pass_success")
        if fps is None:
            retries = et.get("global_retries")
            fps = (retries == 0) if retries is not None else None
        if fps is not None:
            fps_vals.append(int(fps))

        # token utilization — total context tokens injected across all
        # agent-to-agent communication edges (what CUGCR actually compresses).
        # orchestrator_tokens.input measures the AGGREGATE/decomposer calls, which
        # are nearly identical across modes and hide the CUGCR effect entirely.
        tok = sum(
            e.get("context_token_count", 0)
            for e in et.get("communication_edges", [])
        )
        tokens.append(tok)

    return {
        "latency":      np.mean(latencies) if latencies else 0,
        "latency_std":  np.std(latencies)  if latencies else 0,
        "success_rate": np.mean(fps_vals) * 100 if fps_vals else 0,
        "tokens":       np.mean(tokens)   if tokens   else 0,
        "tokens_std":   np.std(tokens)    if tokens   else 0,
        "n":            len(runs),
    }


def print_table(metrics: dict):
    print(f"\n  {'Mode':<10} {'N':>4}  {'Latency(s)':>11}  {'FPS%':>6}  {'Tokens(in)':>11}")
    print(f"  {'─'*10} {'─'*4}  {'─'*11}  {'─'*6}  {'─'*11}")
    for mode in MODES:
        if mode not in metrics:
            continue
        m = metrics[mode]
        print(f"  {mode:<10} {m['n']:>4}  {m['latency']:>9.1f}s  {m['success_rate']:>5.0f}%  {m['tokens']:>11,.0f}")
    print()


def make_chart(metrics: dict, out_path: Path):
    modes   = [m for m in MODES if m in metrics]
    x       = np.arange(len(modes))
    colors  = [COLORS[m] for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.patch.set_facecolor("white")

    panels = [
        ("latency",      "latency_std",  "Mean Latency (s)",            "Latency"),
        ("success_rate", None,           "First-Pass Success Rate (%)",  "Success Rate"),
        ("tokens",       "tokens_std",   "Mean Context Tokens Injected", "Token Utilization"),
    ]

    for ax, (metric, err_key, ylabel, title) in zip(axes, panels):
        vals = [metrics[m][metric] for m in modes]
        errs = [metrics[m][err_key] for m in modes] if err_key else None

        bars = ax.bar(x, vals, 0.5, color=colors, edgecolor="white",
                      yerr=errs, capsize=4, error_kw={"linewidth": 1, "color": "#555"},
                      zorder=3)

        for bar, v in zip(bars, vals):
            label = f"{v:.0f}%" if metric == "success_rate" else (
                    f"{v:.1f}s" if metric == "latency" else f"{v:,.0f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(vals) * 0.02),
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in modes], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(max(vals) * 1.25, 1))
        ax.yaxis.grid(True, color="#eee", zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    n_str = " / ".join(f"{m}=N{metrics[m]['n']}" for m in modes)
    fig.suptitle(f"Pipeline Metrics by Ablation Mode  ({n_str})",
                 fontsize=11, y=1.02)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path,
                   default=ROOT / "raw_telemetry")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "pipeline_metrics.png")
    args = p.parse_args()

    by_mode = load_runs(args.dir)
    if not by_mode:
        print(f"No runs found in {args.dir}")
        sys.exit(1)

    metrics = {mode: compute_metrics(runs) for mode, runs in by_mode.items()}
    print_table(metrics)
    make_chart(metrics, args.out)


if __name__ == "__main__":
    main()
