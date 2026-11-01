"""
plot_self_healing.py — First-pass success rate + self-healing analysis (adaptive mode)
Usage:
    python scripts/eval/plot_self_healing.py
    python scripts/eval/plot_self_healing.py --dir historic_telemetry/raw_telemetry-lite
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


def load_adaptive(tel_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(tel_dir.glob("run_*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("ablation_mode") == "adaptive":
                runs.append(d)
        except Exception:
            pass
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path,
                   default=ROOT / "historic_telemetry" / "raw_telemetry-lite")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "self_healing.png")
    args = p.parse_args()

    runs = load_adaptive(args.dir)
    if not runs:
        print(f"No adaptive runs in {args.dir}")
        sys.exit(1)

    # ── Extract metrics ───────────────────────────────────────────────────────
    by_retries: dict[int, list[float]] = defaultdict(list)
    retry_events = []

    for r in runs:
        et      = r.get("engine_telemetry", {})
        retries = et.get("global_retries", 0) or 0
        score   = et.get("final_judge_score") or 0
        fps     = et.get("first_pass_success")
        if fps is None:
            fps = (retries == 0)
        by_retries[retries].append(float(score))

        for evt in (et.get("retry_events") or []):
            retry_events.append({
                "before":   evt.get("score_before", 0),
                "after":    evt.get("score_after",  0),
                "improved": evt.get("improved", False),
            })

    n_total   = len(runs)
    fps_count = len(by_retries.get(0, []))
    fps_rate  = fps_count / n_total * 100

    print(f"\n  Adaptive runs : {n_total}")
    print(f"  FPS rate      : {fps_rate:.0f}%  ({fps_count}/{n_total})")
    print(f"  Retry events  : {len(retry_events)}")
    if retry_events:
        improved = sum(e["improved"] for e in retry_events)
        print(f"  Retry improved: {improved}/{len(retry_events)}  ({improved/len(retry_events)*100:.0f}%)")
    print()
    for k in sorted(by_retries):
        sc = by_retries[k]
        print(f"  retries={k}: n={len(sc)}  mean_score={np.mean(sc):.1f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("white")

    # Panel 1: outcome distribution + mean score per retry count
    retry_counts = sorted(by_retries.keys())
    counts       = [len(by_retries[k]) for k in retry_counts]
    mean_scores  = [np.mean(by_retries[k]) for k in retry_counts]
    bar_colors   = ["#2ecc71" if k == 0 else "#e67e22" if k == 1 else "#e74c3c"
                    for k in retry_counts]
    labels       = [f"{k} {'retry' if k==1 else 'retries'}" for k in retry_counts]

    bars = ax1.bar(labels, counts, color=bar_colors, edgecolor="white",
                   width=0.5, zorder=3)
    for bar, n, s in zip(bars, counts, mean_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"n={n}\nscore={s:.1f}", ha="center", va="bottom", fontsize=9)

    ax1.set_ylabel("Number of Runs", fontsize=10)
    ax1.set_title("Run Outcome Distribution", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(counts) * 1.4)
    ax1.yaxis.grid(True, color="#eee", zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    fps_text = f"FPS = {fps_rate:.0f}%"
    ax1.text(0.97, 0.97, fps_text, transform=ax1.transAxes,
             ha="right", va="top", fontsize=11, fontweight="bold", color="#2ecc71")

    # Panel 2: self-healing effectiveness (score before vs after per retry)
    if retry_events:
        befores  = [e["before"]  for e in retry_events]
        afters   = [e["after"]   for e in retry_events]
        improved = [e["improved"] for e in retry_events]

        improved_deltas    = [a - b for b, a, i in zip(befores, afters, improved) if i]
        notimproved_deltas = [a - b for b, a, i in zip(befores, afters, improved) if not i]

        n_imp  = sum(improved)
        n_fail = len(improved) - n_imp
        imp_rate = n_imp / len(improved) * 100

        cat_labels = ["Improved", "Not improved"]
        cat_counts = [n_imp, n_fail]
        cat_colors = ["#2ecc71", "#e74c3c"]
        cat_means  = [
            np.mean(improved_deltas)    if improved_deltas    else 0,
            np.mean(notimproved_deltas) if notimproved_deltas else 0,
        ]

        x = np.arange(2)
        bars2 = ax2.bar(x, cat_counts, 0.45, color=cat_colors,
                        edgecolor="white", zorder=3)
        for bar, n, delta in zip(bars2, cat_counts, cat_means):
            sign = "+" if delta >= 0 else ""
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                     f"n={n}\nΔ={sign}{delta:.1f}", ha="center", va="bottom", fontsize=9)

        ax2.set_xticks(x)
        ax2.set_xticklabels(cat_labels, fontsize=10)
        ax2.set_ylabel("Number of Retry Events", fontsize=10)
        ax2.set_title("Self-Healing Effectiveness", fontsize=11, fontweight="bold")
        ax2.set_ylim(0, max(cat_counts) * 1.4)
        ax2.yaxis.grid(True, color="#eee", zorder=0)
        ax2.set_axisbelow(True)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.text(0.97, 0.97, f"Retry success = {imp_rate:.0f}%",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=11, fontweight="bold", color="#2ecc71")
    else:
        ax2.text(0.5, 0.5, "No retry event data available",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)
        ax2.axis("off")

    fig.suptitle(f"Adaptive Mode — Self-Healing Analysis  (N={n_total} runs)",
                 fontsize=11, y=1.02)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n  → {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
