"""
llm_judge_telemetry.py — LLM-as-Judge evaluation over raw telemetry runs
=========================================================================
Loads every run_*.json from raw_telemetry/, sends the final_answer to
OpenAI (Config.JUDGE_MODEL) for multi-dimensional scoring, and produces
a comparison table across ablation modes.

Dimensions scored (total = 10):
  • Correctness  (0–4): Technical/factual accuracy of explanations and code
  • Completeness (0–3): All parts of the query addressed
  • Usability    (0–3): Immediately actionable — a researcher could use this

Additional tracked fields (from telemetry, no LLM call needed):
  • agents_used         — which agent types were invoked
  • first_pass_success  — orchestrator succeeded without retry
  • global_retries      — total retry count
  • latency_s           — wall time in seconds
  • internal_judge_score — score from the engine's own Gemini judge

Usage:
    python scripts/llm_judge_telemetry.py
    python scripts/llm_judge_telemetry.py --dir raw_telemetry --out results/llm_judge_telemetry.json
    python scripts/llm_judge_telemetry.py --dir raw_telemetry_best
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Config

# ---------------------------------------------------------------------------
# Judge prompt — comprehensive, agent-agnostic
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are an expert bioinformatics systems evaluator assessing the output of a multi-agent AI pipeline.

The system was given this query:
"{query}"

The agents invoked were: {agents}

The system produced this final answer:
<answer>
{answer}
</answer>

Score the answer on a scale of 0–10 using EXACTLY these three criteria:

1. **Correctness** (0–4): Are the technical explanations, code, and cloud designs factually accurate?
   - 4: All technical claims correct; code uses proper APIs; cloud architecture is valid
   - 3: Minor inaccuracies or suboptimal choices that wouldn't cause failure
   - 2: Some incorrect API calls, wrong parameters, or flawed logic
   - 1: Significant errors; code/architecture would not work as-is
   - 0: Fundamentally wrong or hallucinated content

2. **Completeness** (0–3): Are ALL components requested in the query present and addressed?
   - 3: Every requested component (explanation + code + cloud design + experts, etc.) is present
   - 2: Most components present; one minor part missing or shallow
   - 1: Only some components addressed; significant gaps
   - 0: Most of the query is unanswered

3. **Usability** (0–3): Could a bioinformatics researcher immediately use this output?
   - 3: Ready to use — code runs, architecture is deployable, experts are real and named
   - 2: Mostly usable with minor edits (fill in a path, add an import)
   - 1: Needs significant work to be usable (missing dependencies, incomplete code)
   - 0: Not usable in current form

Notes:
- If the answer is a refusal or error message, score 0 across all dimensions.
- A "[SHORTFALL]" marker for missing experts is acceptable — do not penalize if experts were genuinely not found.
- Code that is syntactically correct but uses wrong library functions counts as Correctness=1–2.

Return ONLY this JSON (no extra text):
{{
  "correctness": <0–4>,
  "completeness": <0–3>,
  "usability": <0–3>,
  "total": <0–10>,
  "rationale": "<one or two sentences summarising the main strengths and weaknesses>"
}}
"""


async def judge_one(
    client: AsyncOpenAI,
    query: str,
    agents: list[str],
    answer: str,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        query=query,
        agents=", ".join(agents) if agents else "unknown",
        answer=(answer or "")[:4000],
    )
    try:
        result = await client.chat.completions.create(
            model=Config.JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = result.choices[0].message.content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        scores = json.loads(text)
        # Clamp to valid ranges
        scores["correctness"]  = max(0, min(4, int(scores.get("correctness", 0))))
        scores["completeness"] = max(0, min(3, int(scores.get("completeness", 0))))
        scores["usability"]    = max(0, min(3, int(scores.get("usability", 0))))
        scores["total"]        = max(0, min(10, int(scores.get("total", 0))))
        return scores
    except Exception as exc:
        return {
            "correctness": 0, "completeness": 0, "usability": 0,
            "total": 0, "rationale": f"Judge error: {exc}",
        }


def load_runs(telemetry_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(telemetry_dir.glob("run_*.json")):
        try:
            d = json.loads(f.read_text())
            runs.append(d)
        except Exception as exc:
            print(f"  [WARN] Could not load {f.name}: {exc}")
    return runs


def extract_agents(run: dict) -> list[str]:
    tasks = run.get("engine_telemetry", {}).get("initial_tasks", [])
    seen, ordered = set(), []
    for t in tasks:
        a = t.get("agent", "")
        if a and a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered


async def evaluate_all(runs: list[dict], client: AsyncOpenAI) -> list[dict]:
    results = []
    total = len(runs)
    for i, run in enumerate(runs, 1):
        et      = run.get("engine_telemetry", {})
        qid     = run.get("query_id", "?")
        mode    = run.get("ablation_mode", "?")
        query   = run.get("query", et.get("user_query", ""))
        answer  = et.get("final_answer", "") or ""
        agents  = extract_agents(run)

        internal_score = et.get("final_judge_score")
        fps    = et.get("first_pass_success", None)
        retries = et.get("global_retries", 0)
        latency = et.get("latency_seconds", run.get("wall_latency_ms", 0) / 1000)

        print(f"  [{i:02d}/{total}] {qid} | {mode:<9} | agents={agents}")

        scores = await judge_one(client, query, agents, answer)
        passed = scores["total"] >= 7

        print(
            f"           correct={scores['correctness']}/4  "
            f"complete={scores['completeness']}/3  "
            f"usable={scores['usability']}/3  "
            f"total={scores['total']}/10  "
            f"{'PASS' if passed else 'FAIL'}"
        )
        print(f"           {scores['rationale']}")

        results.append({
            "run_id":               run.get("run_id", ""),
            "query_id":             qid,
            "ablation_mode":        mode,
            "query":                query,
            "agents_used":          agents,
            "latency_s":            round(latency, 2),
            "first_pass_success":   fps,
            "global_retries":       retries,
            "internal_judge_score": internal_score,
            "judge":                scores,
            "passed":               passed,
        })

    return results


def print_summary(results: list[dict]):
    modes = sorted({r["ablation_mode"] for r in results})

    print(f"\n── Per-Mode Summary ─────────────────────────────────────────")
    print(f"  {'Mode':<12} {'Runs':>5} {'Pass%':>6} {'Mean':>6} {'Correct':>8} {'Complete':>9} {'Usable':>7} {'Lat(s)':>7} {'FPS%':>6} {'Retries':>8}")
    print(f"  {'─'*12} {'─'*5} {'─'*6} {'─'*6} {'─'*8} {'─'*9} {'─'*7} {'─'*7} {'─'*6} {'─'*8}")

    for mode in modes:
        rs = [r for r in results if r["ablation_mode"] == mode]
        n  = len(rs)
        if n == 0:
            continue
        pass_rate  = sum(r["passed"] for r in rs) / n * 100
        mean_total = sum(r["judge"]["total"] for r in rs) / n
        mean_corr  = sum(r["judge"]["correctness"] for r in rs) / n
        mean_comp  = sum(r["judge"]["completeness"] for r in rs) / n
        mean_use   = sum(r["judge"]["usability"] for r in rs) / n
        mean_lat   = sum(r["latency_s"] for r in rs) / n
        fps_vals   = [r["first_pass_success"] for r in rs if r["first_pass_success"] is not None]
        fps_rate   = sum(fps_vals) / len(fps_vals) * 100 if fps_vals else 0
        mean_retry = sum(r["global_retries"] or 0 for r in rs) / n
        print(
            f"  {mode:<12} {n:>5} {pass_rate:>5.0f}% {mean_total:>6.2f} "
            f"{mean_corr:>7.2f}/4 {mean_comp:>8.2f}/3 {mean_use:>6.2f}/3 "
            f"{mean_lat:>7.1f} {fps_rate:>5.0f}% {mean_retry:>8.2f}"
        )

    print(f"\n── Per-Query Breakdown ──────────────────────────────────────")
    qids = sorted({r["query_id"] for r in results})
    for qid in qids:
        rs = [r for r in results if r["query_id"] == qid]
        agents_str = "+".join(rs[0]["agents_used"]) if rs else "?"
        query_preview = (rs[0]["query"][:55] + "...") if rs else ""
        print(f"\n  {qid}  [{agents_str}]  {query_preview}")
        print(f"  {'Mode':<12} {'Score':>6} {'Corr':>5} {'Comp':>5} {'Use':>5} {'Pass':>5}  Internal  Rationale")
        print(f"  {'─'*12} {'─'*6} {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*8}  {'─'*35}")
        for r in sorted(rs, key=lambda x: x["ablation_mode"]):
            internal = str(r["internal_judge_score"]) if r["internal_judge_score"] is not None else "?"
            print(
                f"  {r['ablation_mode']:<12} {r['judge']['total']:>6} "
                f"{r['judge']['correctness']:>5} {r['judge']['completeness']:>5} "
                f"{r['judge']['usability']:>5} {'✓' if r['passed'] else '✗':>5}  "
                f"{internal:>8}  {r['judge']['rationale'][:50]}"
            )

    # Internal vs external judge agreement
    paired = [(r["internal_judge_score"], r["judge"]["total"])
              for r in results if r["internal_judge_score"] is not None]
    if paired:
        diffs = [abs(a - b) for a, b in paired]
        mean_diff = sum(diffs) / len(diffs)
        agree = sum(1 for d in diffs if d <= 1) / len(diffs) * 100
        print(f"\n── Internal (Gemini) vs External (OpenAI) Judge ─────────────")
        print(f"  Mean absolute difference : {mean_diff:.2f} pts")
        print(f"  Agreement within 1pt     : {agree:.0f}%")


async def main(telemetry_dir: Path, out_path: Path):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    client = AsyncOpenAI(api_key=api_key)

    print(f"\n── LLM-as-Judge Telemetry Evaluation ────────────────────────")
    print(f"  Directory : {telemetry_dir}")
    print(f"  Judge     : {Config.JUDGE_MODEL}")

    runs = load_runs(telemetry_dir)
    if not runs:
        print("  No run_*.json files found.")
        return

    print(f"  Runs found: {len(runs)}\n")
    results = await evaluate_all(runs, client)
    print_summary(results)

    # Save
    modes = sorted({r["ablation_mode"] for r in results})
    summary_by_mode = {}
    for mode in modes:
        rs = [r for r in results if r["ablation_mode"] == mode]
        n  = len(rs)
        summary_by_mode[mode] = {
            "n": n,
            "pass_rate_pct":        round(sum(r["passed"] for r in rs) / n * 100, 1),
            "mean_total":           round(sum(r["judge"]["total"] for r in rs) / n, 2),
            "mean_correctness":     round(sum(r["judge"]["correctness"] for r in rs) / n, 2),
            "mean_completeness":    round(sum(r["judge"]["completeness"] for r in rs) / n, 2),
            "mean_usability":       round(sum(r["judge"]["usability"] for r in rs) / n, 2),
            "mean_latency_s":       round(sum(r["latency_s"] for r in rs) / n, 2),
        }

    out_data = {
        "judge_model":   Config.JUDGE_MODEL,
        "telemetry_dir": str(telemetry_dir),
        "n_runs":        len(results),
        "summary":       summary_by_mode,
        "results":       results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\n  → {out_path}")
    print("Evaluation done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=Path,
        default=ROOT / "raw_telemetry",
        help="Directory containing run_*.json telemetry files",
    )
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results" / "llm_judge_telemetry.json",
        help="Output path for the evaluation JSON",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dir, args.out))


# ── Per-Mode Summary ─────────────────────────────────────────
#   Mode          Runs  Pass%   Mean  Correct  Complete  Usable  Lat(s)   FPS%  Retries
#   ──────────── ───── ────── ────── ──────── ───────── ─────── ─────── ────── ────────
#   adaptive         5    40%   5.60    2.00/4     2.20/3   1.40/3   166.7    40%     0.80
#   none             5    40%   6.60    2.40/4     2.40/3   1.80/3   139.7    60%     0.40
#   raw              5    40%   6.40    2.40/4     2.20/3   1.80/3   168.7    40%     0.80

# ── Per-Query Breakdown ──────────────────────────────────────

#   BQ01  [rag+code+cloud]  Explain the statistical model behind DESeq2 for differe...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          4     2     1     1     ✗         8  The statistical explanation is mostly accurate at 
#   none              5     2     2     1     ✗         8  The DESeq2 statistical overview is broadly correct
#   raw               4     2     1     1     ✗         6  The answer gives a broadly correct high-level desc

#   BQ02  [scholar+code]  Find 3 experts on Leiden community detection in single-...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          4     1     2     1     ✗         3  The pandas code is syntactically correct, but the 
#   none              9     3     3     3     ✓         7  The answer provides three real experts, valid Pyth
#   raw               7     2     3     2     ✓         3  The Python code is syntactically correct and does 

#   BQ03  [rag+code+cloud]  Explain how to process a whole-genome sequencing datase...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          5     2     2     1     ✗         4  The conceptual GATK explanation is mostly sound an
#   none              5     2     2     1     ✗         8  The conceptual explanation of BQSR and HaplotypeCa
#   raw               5     2     2     1     ✗         7  The conceptual GATK explanation is broadly accurat

#   BQ04  [rag+scholar]  Find 4 experts who have published papers on 'STAR RNA-s...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          7     2     3     2     ✓         3  The answer provides four real researchers tied to 
#   none              6     2     2     2     ✗         4  The answer does identify four real authors associa
#   raw               6     2     2     2     ✗         3  The answer does provide four real authors associat

#   BQ05  [rag+code]  Write a simple Python script using pysam to open a BAM ...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          8     3     3     2     ✓         9  The answer provides a valid pysam script that open
#   none              8     3     3     2     ✓         7  The answer provides a valid pysam script that open
#   raw              10     4     3     3     ✓         8  The answer provides a correct pysam script that op

# ── Internal (Gemini) vs External (OpenAI) Judge ─────────────
#   Mean absolute difference : 2.33 pts
#   Agreement within 1pt     : 27%


# ------------------- AFTER CALIBRATION RUNS ─────────────────────────────────────────
# ── Per-Mode Summary ─────────────────────────────────────────
#   Mode          Runs  Pass%   Mean  Correct  Complete  Usable  Lat(s)   FPS%  Retries
#   ──────────── ───── ────── ────── ──────── ───────── ─────── ─────── ────── ────────
#   adaptive         5    20%   5.60    2.00/4     2.20/3   1.40/3   185.2    40%     0.60
#   none             5    20%   5.80    2.20/4     2.00/3   1.60/3   183.9    40%     1.00

# ── Per-Query Breakdown ──────────────────────────────────────

#   BQ01  [rag+code+cloud]  Explain the statistical model behind DESeq2 for differe...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          5     2     2     1     ✗         8  The DESeq2 explanation is broadly reasonable, but 
#   none              4     1     2     1     ✗         4  The answer covers the requested themes, but it con

#   BQ02  [scholar+code]  Find 3 experts on Leiden community detection in single-...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          5     1     2     2     ✗         3  The Python code is syntactically valid and does cr
#   none              5     2     1     2     ✗         4  The Python dataframe code is syntactically fine, b

#   BQ03  [rag+code+cloud]  Explain how to process a whole-genome sequencing datase...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          5     2     2     1     ✗         6  The answer gives a broadly correct high-level desc
#   none              5     2     2     1     ✗         8  The GATK concepts and core Snakemake steps are mos

#   BQ04  [rag+scholar]  Find 4 experts who have published papers on 'STAR RNA-s...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          4     1     2     1     ✗         3  The answer does provide four named researchers, bu
#   none              6     2     2     2     ✗         6  The answer correctly identifies the seminal STAR p

#   BQ05  [rag+code]  Write a simple Python script using pysam to open a BAM ...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          9     4     3     2     ✓         9  The code correctly uses pysam.AlignmentFile to ope
#   none              9     4     3     2     ✓         9  The answer provides a correct pysam script that op

# ── Internal (Gemini) vs External (OpenAI) Judge ─────────────
#   Mean absolute difference : 1.10 pts
#   Agreement within 1pt     : 70%

# ----------------- After 2nd calibration runs ─────────────────────────────────────────
# ── Per-Mode Summary ─────────────────────────────────────────
#   Mode          Runs  Pass%   Mean  Correct  Complete  Usable  Lat(s)   FPS%  Retries
#   ──────────── ───── ────── ────── ──────── ───────── ─────── ─────── ────── ────────
#   adaptive         5    20%   5.60    2.00/4     2.00/3   1.60/3   231.6    20%     0.80
#   none             5    20%   6.20    2.20/4     2.20/3   1.80/3   157.4    60%     0.60

# ── Per-Query Breakdown ──────────────────────────────────────

#   BQ01  [rag+code+cloud]  Explain the statistical model behind DESeq2 for differe...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          4     1     2     1     ✗         6  The answer gives a broadly reasonable high-level d
#   none              5     2     2     1     ✗         8  The DESeq2 statistical explanation is mostly accur

#   BQ02  [scholar+code]  Find 3 experts on Leiden community detection in single-...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          6     2     2     2     ✗         4  The pandas code is syntactically correct and print
#   none              6     2     2     2     ✗         4  The pandas code is syntactically correct and would

#   BQ03  [rag+code+cloud]  Explain how to process a whole-genome sequencing datase...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          5     2     2     1     ✗         6  The biological explanation and core GATK commands 
#   none              5     2     2     1     ✗         8  The conceptual GATK explanation and core Snakemake

#   BQ04  [rag+scholar]  Find 4 experts who have published papers on 'STAR RNA-s...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive          3     1     1     1     ✗         3  The answer includes four real researchers and a re
#   none              6     2     2     2     ✗         6  The answer does identify four real experts associa

#   BQ05  [code]  Write a simple Python script using pysam to open a BAM ...
#   Mode          Score  Corr  Comp   Use  Pass  Internal  Rationale
#   ──────────── ────── ───── ───── ───── ─────  ────────  ───────────────────────────────────
#   adaptive         10     4     3     3     ✓        10  The answer provides a correct pysam script that op
#   none              9     3     3     3     ✓         9  The core pysam script is correct, syntactically va

# ── Internal (Gemini) vs External (OpenAI) Judge ─────────────
#   Mean absolute difference : 1.30 pts
#   Agreement within 1pt     : 50%