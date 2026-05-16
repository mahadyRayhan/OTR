"""
judge_ablation.py — LLM-as-Judge: none / raw / adaptive, 6-dimension scoring + bar chart
Loads from historic_telemetry/raw_telemetry_best by default (30 runs × 3 modes).
Runs judge calls concurrently to minimise wall time.

Usage:
    python scripts/eval/judge_none_vs_raw.py
    python scripts/eval/judge_none_vs_raw.py --dir historic_telemetry/raw_telemetry_best
    python scripts/eval/judge_none_vs_raw.py --concurrency 10
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from config import Config

MODES  = ["none", "raw", "adaptive"]
COLORS = {"none": "#3498db", "raw": "#e67e22", "adaptive": "#2ecc71"}
DIMS   = ["Correctness", "Completeness", "Relevance", "Grounding", "Specificity", "Coherence"]

JUDGE_PROMPT = """You are evaluating the output of a multi-agent bioinformatics AI pipeline.

Query: "{query}"
Answer: "{answer}"

Score each dimension from 0 to 10:
- Correctness  : technical/factual accuracy (0=wrong, 10=fully correct)
- Completeness : all parts of the query addressed (0=missing most, 10=fully addressed)
- Relevance    : answer stays on-topic (0=off-topic, 10=fully relevant)
- Grounding    : claims backed by real tools/methods, not hallucinated (0=hallucinated, 10=well-grounded)
- Specificity  : concrete details — tool names, parameters, commands (0=vague, 10=highly specific)
- Coherence    : logical structure and clarity (0=incoherent, 10=clear and well-structured)

Return ONLY this JSON:
{{
  "correctness": <0-10>,
  "completeness": <0-10>,
  "relevance": <0-10>,
  "grounding": <0-10>,
  "specificity": <0-10>,
  "coherence": <0-10>,
  "rationale": "<one sentence>"
}}
"""


async def judge(client, query, answer, sem):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=Config.JUDGE_MODEL,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    query=query, answer=(answer or "")[:4000])}],
                temperature=0.0,
            )
            text = r.choices[0].message.content.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0].strip().lstrip("json")
            scores = json.loads(text)
            for d in DIMS:
                scores[d.lower()] = max(0, min(10, int(scores.get(d.lower(), 0))))
            return scores
        except Exception as e:
            return {d.lower(): 0 for d in DIMS} | {"rationale": str(e)}


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


def print_table(results: list[dict]):
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r["scores"])

    print(f"\n  {'Mode':<10}  " + "  ".join(f"{d[:5]:>6}" for d in DIMS))
    print(f"  {'─'*10}  " + "  ".join("─"*6 for _ in DIMS))
    for mode in MODES:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        means = {d.lower(): sum(r[d.lower()] for r in rs) / len(rs) for d in DIMS}
        print(f"  {mode:<10}  " + "  ".join(f"{means[d.lower()]:>6.1f}" for d in DIMS))
    print()


def make_chart(results: list[dict], out_path: Path):
    by_mode = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r["scores"])

    modes_present = [m for m in MODES if m in by_mode]
    n_modes = len(modes_present)
    x = np.arange(len(DIMS))
    w = 0.22

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("white")

    for i, mode in enumerate(modes_present):
        rs   = by_mode[mode]
        vals = [sum(r[d.lower()] for r in rs) / len(rs) for d in DIMS]
        errs = [np.std([r[d.lower()] for r in rs]) for d in DIMS]
        offset = (i - (n_modes - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=f"{mode.capitalize()} (n={len(rs)})",
                      color=COLORS[mode], edgecolor="white", zorder=3,
                      yerr=errs, capsize=3, error_kw={"linewidth": 0.8, "color": "#555"})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(DIMS, fontsize=10)
    ax.set_ylim(0, 12)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("Mean Score (0–10)", fontsize=10)
    ax.set_title("LLM-as-Judge: None vs Raw vs Adaptive — 6 Dimensions",
                 fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, color="#eee", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  → {out_path}")
    plt.close(fig)


async def main(tel_dir: Path, out_json: Path, out_fig: Path, concurrency: int):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem    = asyncio.Semaphore(concurrency)

    by_mode = load_runs(tel_dir)
    total   = sum(len(v) for v in by_mode.values())
    print(f"\n  {total} runs across {list(by_mode.keys())}  ({tel_dir.name})")
    print(f"  Judging with concurrency={concurrency} ...\n")

    # Build all tasks
    tasks, meta = [], []
    for mode in MODES:
        for run in by_mode.get(mode, []):
            et     = run.get("engine_telemetry", {})
            query  = run.get("query", et.get("user_query", ""))
            answer = et.get("final_answer", "")
            tasks.append(judge(client, query, answer, sem))
            meta.append({"query_id": run.get("query_id", "?"), "mode": mode, "query": query})

    scores_list = await asyncio.gather(*tasks)

    results = []
    for m, scores in zip(meta, scores_list):
        results.append({**m, "scores": scores})
        print(f"  {m['query_id']} | {m['mode']:<8} | " +
              " ".join(f"{d[:4]}={scores[d.lower()]}" for d in DIMS))

    print_table(results)
    make_chart(results, out_fig)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"  → {out_json}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path,
                   default=ROOT / "raw_telemetry")
    p.add_argument("--out", type=Path,
                   default=ROOT / "results" / "judge_ablation.json")
    p.add_argument("--fig", type=Path,
                   default=ROOT / "results" / "judge_ablation_chart.png")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Number of parallel judge calls")
    args = p.parse_args()
    asyncio.run(main(args.dir, args.out, args.fig, args.concurrency))
