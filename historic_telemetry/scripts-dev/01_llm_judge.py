"""
01_llm_judge.py — System-Level Multi-Dimensional LLM-as-Judge Evaluation
=========================================================================
For each run in raw_telemetry/, sends the (query, final_answer) pair to an
LLM judge with a structured rubric that scores 8 quality dimensions
independently (all 1-10). Results are cached in eval_cache/ so the API is
only called once per run.

Dimensions evaluated
--------------------
  completeness      — Did the response address EVERY part of the query?
  helpfulness       — Is the answer practically useful for the user's task?
  factual_accuracy  — Are scientific / technical claims correct?
  coherence         — Is the response well-structured and logically organised?
  relevance         — Does it stay on topic without padding or tangents?
  code_quality      — Is the generated code syntactically correct and runnable?
                      (auto-scored 10 if no code was requested)
  citation_grounding— Are claims tied to specific papers / sources?
  specificity       — Concrete details rather than vague generalities?

Run:   python scripts/eval/01_llm_judge.py
Input: raw_telemetry/run_*.json
Cache: eval_cache/01_llm_judge_scores.json   (re-run safe — skips scored runs)
Out:   figures/01_llm_judge.{png,pdf}
"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
TELEMETRY_DIR = ROOT / "raw_telemetry"
CACHE_DIR     = ROOT / "eval_cache"
FIG_DIR       = ROOT / "figures"
CACHE_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

CACHE_FILE    = CACHE_DIR / "01_llm_judge_scores.json"

# ── Model ──────────────────────────────────────────────────────────────────
JUDGE_MODEL   = os.getenv("JUDGE_MODEL", "gpt-5.4")   # override via env if needed
API_KEY       = os.getenv("OPENAI_API_KEY", "")
CONCURRENCY   = 5          # parallel API calls
RETRY_DELAY   = 2.0        # seconds between retries on rate-limit errors

# ── Dimensions ─────────────────────────────────────────────────────────────
DIMENSIONS = [
    "completeness",
    "helpfulness",
    "factual_accuracy",
    "coherence",
    "relevance",
    "code_quality",
    "citation_grounding",
    "specificity",
    "dependency_satisfaction",   # KEY metric: did the answer use upstream agent outputs?
]

DIM_LABELS = {
    "completeness":          "Completeness",
    "helpfulness":           "Helpfulness",
    "factual_accuracy":      "Factual\nAccuracy",
    "coherence":             "Coherence",
    "relevance":             "Relevance",
    "code_quality":          "Code\nQuality",
    "citation_grounding":    "Citation\nGrounding",
    "specificity":           "Specificity",
    "dependency_satisfaction": "Dependency\nSatisfaction",
    "overall":               "Overall\nAvg",
}

# ── Query-group membership ─────────────────────────────────────────────────
GROUPS = {
    "G1\nFull-Stack":  {f"BQ{i:02d}" for i in range(1,  6)},
    "G2\nCoref.":      {f"BQ{i:02d}" for i in range(6,  11)},
    "G3\nStress":      {f"BQ{i:02d}" for i in range(11, 16)},
    "G4\nMissing":     {f"BQ{i:02d}" for i in range(16, 21)},
    "G5\nAdvers.":     {f"BQ{i:02d}" for i in range(21, 26)},
    "G6\nBaseline":    {f"BQ{i:02d}" for i in range(26, 31)},
}
QID_TO_GROUP = {qid: g for g, qids in GROUPS.items() for qid in qids}
MODES        = ["none", "raw", "adaptive"]

# ── Visual style ───────────────────────────────────────────────────────────
STYLES = {
    "none":     {"face": "#d9d9d9", "edge": "#4d4d4d", "hatch": "xx",  "label": "NONE"},
    "raw":      {"face": "#ffcc99", "edge": "#cc3300", "hatch": "//",  "label": "RAW"},
    "adaptive": {"face": "#99ffcc", "edge": "#006633", "hatch": "..",  "label": "ADAPTIVE (Ours)"},
}

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         20,
    "font.weight":       "bold",
    "axes.labelsize":    36,
    "axes.titlesize":    40,
    "xtick.labelsize":   26,
    "ytick.labelsize":   30,
    "axes.labelweight":  "bold",
    "axes.titleweight":  "bold",
    "legend.fontsize":   28,
    "hatch.linewidth":   2.0,
})
sns.set_style("whitegrid", {"grid.linestyle": "--"})

# ── Structured judge prompt ────────────────────────────────────────────────
# Two prompt variants:
#   JUDGE_PROMPT_WITH_CONTEXT — used when upstream agent outputs are available
#     (RAW / ADAPTIVE modes). The judge can verify dependency satisfaction.
#   JUDGE_PROMPT_NO_CONTEXT   — used when no inter-agent context existed
#     (NONE mode). dependency_satisfaction is auto-scored 1 because by design
#     no upstream findings could have been used.
# ──────────────────────────────────────────────────────────────────────────

_RUBRIC_SHARED = """\
SCORING RUBRIC (score each 1–10, independently)
------------------------------------------------
completeness         : Did the response address EVERY sub-task in the query?
                       Deduct heavily if any requested deliverable (code,
                       expert names, cloud architecture, explanation) is absent.
helpfulness          : Is the response practically useful for the user's task?
                       Would a working bioinformatics researcher find this
                       actionable and ready to use?
factual_accuracy     : Are the scientific and technical claims correct?
                       Penalise hallucinated algorithm details, wrong API calls,
                       or non-existent paper citations.
coherence            : Is the response well-structured, logically ordered, and
                       free from internal contradictions?
relevance            : Does it stay strictly on topic?
                       Penalise generic filler, tangents, or off-topic content.
code_quality         : If code is present — is it syntactically valid, uses the
                       correct library APIs, and would run without modification?
                       Auto-score 10 if no code was requested.
citation_grounding   : Are factual claims backed by specific paper titles,
                       author names, or identifiable sources?
                       "Studies show …" with no attribution scores low.
specificity          : Does the response give concrete, precise details
                       (exact parameter values, algorithm steps, specific AWS
                       resource types) rather than vague generalities?
"""

JUDGE_PROMPT_WITH_CONTEXT = (
    "You are an expert evaluator for a multi-agent AI research system "
    "specialising in bioinformatics. The system is composed of specialised "
    "agents (RAG, Scholar, Code, Cloud) whose outputs are passed between them. "
    "A user posed a complex query; the system ran its agents and produced a "
    "final response. You are given the ACTUAL outputs that each upstream agent "
    "produced so you can verify whether the final answer correctly used them.\n\n"
    + _RUBRIC_SHARED
    + """\
dependency_satisfaction: CRITICAL — this is the key metric for a multi-agent system.
                         You are shown WHAT each upstream agent actually retrieved or
                         generated. Score how well the final answer USED those specific
                         findings rather than substituting generic LLM knowledge.
                         Score 10  → every specific fact / name / code artefact from
                                     upstream agents appears correctly in the answer.
                         Score 5-7 → partial use; some upstream content present,
                                     some replaced with plausible-sounding fabrications.
                         Score 1-3 → the answer ignores or contradicts upstream findings;
                                     uses hallucinated substitutes instead.

USER QUERY:
{query}

WHAT UPSTREAM AGENTS ACTUALLY PRODUCED (ground truth for dependency check):
{upstream_context}

FINAL SYSTEM RESPONSE:
{answer}

Reply with ONLY a valid JSON object — no markdown, no explanation outside JSON:
{{
  "completeness":           <int 1-10>,
  "helpfulness":            <int 1-10>,
  "factual_accuracy":       <int 1-10>,
  "coherence":              <int 1-10>,
  "relevance":              <int 1-10>,
  "code_quality":           <int 1-10>,
  "citation_grounding":     <int 1-10>,
  "specificity":            <int 1-10>,
  "dependency_satisfaction":<int 1-10>,
  "reasoning": "<one sentence: main weakness of the final answer>"
}}
"""
)

JUDGE_PROMPT_NO_CONTEXT = (
    "You are an expert evaluator for a multi-agent AI research system "
    "specialising in bioinformatics. In this run the system operated in "
    "BLIND mode: each agent worked in isolation with NO inter-agent context "
    "passing. Therefore dependency_satisfaction is automatically scored 1 "
    "(agents could not have used each other's findings by design). "
    "Evaluate all other 8 dimensions independently.\n\n"
    + _RUBRIC_SHARED
    + """\
USER QUERY:
{query}

FINAL SYSTEM RESPONSE:
{answer}

Reply with ONLY a valid JSON object — no markdown, no explanation outside JSON:
{{
  "completeness":           <int 1-10>,
  "helpfulness":            <int 1-10>,
  "factual_accuracy":       <int 1-10>,
  "coherence":              <int 1-10>,
  "relevance":              <int 1-10>,
  "code_quality":           <int 1-10>,
  "citation_grounding":     <int 1-10>,
  "specificity":            <int 1-10>,
  "dependency_satisfaction": 1,
  "reasoning": "<one sentence: main weakness of the final answer>"
}}
"""
)

# ── Build upstream context string from telemetry ───────────────────────────

def build_upstream_context(record: dict) -> str:
    """
    Extracts per-agent outputs from got_dag_nodes (non-aggregate nodes)
    and formats them as a readable ground-truth block for the judge.
    Returns empty string if no upstream data is available.
    """
    nodes = record.get("engine_telemetry", {}).get("got_dag_nodes", [])
    parts = []
    for node in nodes:
        op     = node.get("operator", "")
        agent  = node.get("agent", "?").upper()
        output = (node.get("output") or "").strip()
        if op in ("aggregate",) or not output:
            continue
        parts.append(
            f"[{agent} AGENT — op: {op}]\n{output[:600]}"
        )
    return "\n\n---\n\n".join(parts)


# ── API evaluation ─────────────────────────────────────────────────────────

async def evaluate_one(
    client:           AsyncOpenAI,
    run_id:           str,
    query:            str,
    answer:           str,
    upstream_context: str,
    ablation_mode:    str,
    sem:              asyncio.Semaphore,
) -> dict | None:
    # NONE mode: agents ran in isolation — dependency_satisfaction is fixed at 1
    has_upstream = bool(upstream_context.strip()) and ablation_mode != "none"

    if has_upstream:
        prompt = JUDGE_PROMPT_WITH_CONTEXT.format(
            query=query[:2000],
            upstream_context=upstream_context[:3000],
            answer=answer[:3500],
        )
    else:
        prompt = JUDGE_PROMPT_NO_CONTEXT.format(
            query=query[:2000],
            answer=answer[:4000],
        )
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content.strip()
                data = json.loads(raw)
                # Validate all dimensions present and in range
                scores = {}
                for dim in DIMENSIONS:
                    val = data.get(dim)
                    # NONE mode: dependency_satisfaction is hard-coded to 1 in the prompt;
                    # the model may echo it back as an int or omit it — handle both.
                    if val is None and dim == "dependency_satisfaction" and not has_upstream:
                        scores[dim] = 1
                    elif val is None:
                        raise ValueError(f"missing dimension '{dim}'")
                    else:
                        scores[dim] = max(1, min(10, int(val)))
                scores["reasoning"]       = str(data.get("reasoning", ""))
                scores["had_upstream"]    = has_upstream
                scores["run_id"]          = run_id
                return scores
            except Exception as exc:
                if attempt < 2:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"  [FAIL] run {run_id}: {exc}")
                    return None


async def run_evaluation(records: list[dict], existing: dict) -> dict:
    if not API_KEY:
        print("[ERROR] OPENAI_API_KEY not set. Cannot run LLM evaluation.")
        sys.exit(1)

    client  = AsyncOpenAI(api_key=API_KEY)
    sem     = asyncio.Semaphore(CONCURRENCY)
    results = dict(existing)

    # Re-score if dependency_satisfaction is missing (handles cache from old version)
    def needs_scoring(r: dict) -> bool:
        cached = results.get(r.get("run_id"))
        if cached is None:
            return True
        if "dependency_satisfaction" not in cached:
            return True
        return False

    to_score = [
        r for r in records
        if needs_scoring(r)
        and r.get("engine_telemetry", {}).get("final_answer")
    ]

    if not to_score:
        print("All runs already scored — using cache.")
        return results

    print(f"Scoring {len(to_score)} runs with {JUDGE_MODEL} "
          f"({CONCURRENCY} concurrent)...")

    tasks = [
        evaluate_one(
            client,
            r["run_id"],
            r.get("query", r.get("engine_telemetry", {}).get("user_query", "")),
            r["engine_telemetry"]["final_answer"],
            build_upstream_context(r),
            r.get("ablation_mode", ""),
            sem,
        )
        for r in to_score
    ]

    scored = await asyncio.gather(*tasks)

    for rec, score in zip(to_score, scored):
        if score is not None:
            results[rec["run_id"]] = {
                "ablation_mode": rec.get("ablation_mode", ""),
                "query_id":      rec.get("query_id", ""),
                **score,
            }

    # Persist cache
    with open(CACHE_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Scores cached → {CACHE_FILE}")
    return results


# ── Statistics ─────────────────────────────────────────────────────────────

def compute_stats(scores: dict) -> dict:
    """
    Returns {mode: {dimension: (mean, sem, n)}}
    Also adds 'overall' = mean of all DIMENSIONS per run.
    """
    buckets: dict = defaultdict(lambda: defaultdict(list))

    for entry in scores.values():
        mode = entry.get("ablation_mode", "")
        if mode not in MODES:
            continue
        for dim in DIMENSIONS:
            val = entry.get(dim)
            if val is not None:
                buckets[mode][dim].append(float(val))
        # overall = mean across all 8 dimensions for this run
        vals = [entry[d] for d in DIMENSIONS if d in entry]
        if vals:
            buckets[mode]["overall"].append(np.mean(vals))

    stats: dict = {}
    for mode in MODES:
        stats[mode] = {}
        for dim in DIMENSIONS + ["overall"]:
            v = buckets[mode].get(dim, [])
            if v:
                stats[mode][dim] = (
                    np.mean(v),
                    np.std(v, ddof=1) / np.sqrt(len(v)),
                    len(v),
                )
            else:
                stats[mode][dim] = (np.nan, 0.0, 0)
    return stats


# ── Main bar chart ─────────────────────────────────────────────────────────

def plot_bar(stats: dict, out_dir: Path) -> None:
    all_dims   = DIMENSIONS + ["overall"]
    x_labels   = [DIM_LABELS[d] for d in all_dims]
    n_groups   = len(all_dims)
    bar_width  = 0.25
    x          = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(28, 12))

    for i, mode in enumerate(MODES):
        s      = STYLES[mode]
        offset = (i - 1) * bar_width
        means  = [stats[mode][d][0] for d in all_dims]
        errs   = [stats[mode][d][1] for d in all_dims]

        bars = ax.bar(
            x + offset, means, bar_width,
            color=s["face"], edgecolor=s["edge"], linewidth=2,
            hatch=s["hatch"], label=s["label"],
            yerr=errs, capsize=5,
            error_kw={"elinewidth": 1.8, "ecolor": s["edge"], "capthick": 2},
            zorder=3,
        )

        for bar, mean in zip(bars, means):
            if np.isnan(mean) or mean == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.18,
                f"{mean:.1f}",
                ha="center", va="bottom",
                color=s["edge"], fontsize=19, fontweight="bold",
            )

    # Separator before Overall
    ax.axvline(n_groups - 1 - 0.5, color="#888888",
               linestyle="--", linewidth=1.8, zorder=2)

    ax.set_ylabel("Score (1–10)", labelpad=12)
    ax.set_title("LLM-as-Judge: Multi-Dimensional Quality Evaluation\n"
                 "by Ablation Mode (mean ± SEM)", pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, ha="center")
    ax.set_ylim(0, 13)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    sns.despine()

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )

    plt.tight_layout()
    _save(fig, out_dir, "01_llm_judge_bar")


# ── Radar chart (supplementary) ────────────────────────────────────────────

def plot_radar(stats: dict, out_dir: Path) -> None:
    dims   = DIMENSIONS  # 8 axes
    labels = [DIM_LABELS[d].replace("\n", " ") for d in dims]
    N      = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    radar_colors = {
        "none":     ("#4d4d4d", "--"),
        "raw":      ("#cc3300", "-."),
        "adaptive": ("#006633", "-"),
    }

    fig, ax = plt.subplots(figsize=(12, 12),
                           subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for mode in MODES:
        s      = STYLES[mode]
        color, ls = radar_colors[mode]
        vals   = [stats[mode][d][0] for d in dims]
        vals  += vals[:1]
        ax.plot(angles, vals, ls, color=color, linewidth=2.5,
                label=s["label"])
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=22, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=18)
    ax.set_title("LLM-as-Judge: Quality Profile by Mode",
                 fontsize=34, fontweight="bold", pad=28)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=26,
    )

    plt.tight_layout()
    _save(fig, out_dir, "01_llm_judge_radar")


# ── Helpers ────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight",
                    format=ext if ext == "pdf" else None)
        print(f"Saved {path}")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    # Re-apply after reset so subsequent plots stay styled
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         20,
        "font.weight":       "bold",
        "axes.labelsize":    36,
        "axes.titlesize":    40,
        "xtick.labelsize":   26,
        "ytick.labelsize":   30,
        "axes.labelweight":  "bold",
        "axes.titleweight":  "bold",
        "legend.fontsize":   28,
        "hatch.linewidth":   2.0,
    })
    sns.set_style("whitegrid", {"grid.linestyle": "--"})


def load_records(directory: Path) -> list[dict]:
    records = []
    for f in sorted(directory.glob("run_*.json")):
        try:
            with open(f) as fh:
                records.append(json.load(fh))
        except Exception as exc:
            print(f"  [WARN] skipping {f.name}: {exc}")
    print(f"Loaded {len(records)} run records.")
    return records


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as fh:
            data = json.load(fh)
        print(f"Cache: {len(data)} existing scores loaded.")
        return data
    return {}


# ── Summary table ──────────────────────────────────────────────────────────

def print_table(stats: dict) -> None:
    col_w = 16
    all_dims = DIMENSIONS + ["overall"]

    print("\n── LLM-as-Judge Scores: mean ± SEM (scale 1–10) ──")
    header = f"{'Dimension':<22}" + "".join(
        f"{'NONE':>{col_w}}  {'RAW':>{col_w}}  {'ADAPTIVE':>{col_w}}  {'Δ(A-N)':>8}   "
    )
    print(f"{'Dimension':<22}  {'NONE':>{col_w}}  {'RAW':>{col_w}}  {'ADAPTIVE':>{col_w}}  {'Δ(A−N)':>8}")
    print("─" * 90)

    for dim in all_dims:
        label = DIM_LABELS[dim].replace("\n", " ")

        def fmt(mode: str) -> str:
            m, s, n = stats[mode].get(dim, (None, 0, 0))
            return f"{m:5.2f}±{s:.2f}" if (m is not None and not np.isnan(m)) else "   N/A   "

        m_n = stats["none"]["adaptive" if False else dim][0]     # none mean
        m_a = stats["adaptive"][dim][0]                           # adaptive mean
        m_n_val = stats["none"][dim][0]
        delta = f"{m_a - m_n_val:+.2f}" if not (np.isnan(m_a) or np.isnan(m_n_val)) else " N/A"

        print(f"{label:<22}  {fmt('none'):>{col_w}}  {fmt('raw'):>{col_w}}  {fmt('adaptive'):>{col_w}}  {delta:>8}")


# ── Entry point ────────────────────────────────────────────────────────────

async def main() -> None:
    if not TELEMETRY_DIR.exists():
        print(f"[ERROR] Telemetry dir not found: {TELEMETRY_DIR}")
        sys.exit(1)

    records  = load_records(TELEMETRY_DIR)
    cache    = load_cache()
    scores   = await run_evaluation(records, cache)

    if not scores:
        print("[ERROR] No scores available. Check API key and run files.")
        sys.exit(1)

    stats = compute_stats(scores)
    print_table(stats)
    plot_bar(stats, FIG_DIR)
    plot_radar(stats, FIG_DIR)
    print("\nDone. Output in figures/")


if __name__ == "__main__":
    asyncio.run(main())
