"""
run_judge_ablation.py
=====================
Runs the Judge Ablation study: takes the top-N adaptive queries from
raw_telemetry_best (ranked by mean DAG node quality), re-executes each
through the full adaptive pipeline with BOTH LLM judges disabled, and
saves telemetry to raw_telemetry_no_judge/.

What "no judge" means:
  - Agent-level judge (Gemini): disabled → heuristic judge used instead
    (length + structural markers). No LLM-driven retries or expansion.
  - Final answer judge (GPT):   disabled → skipped entirely.

Compare the output against raw_telemetry_best (both judges enabled) using:
    python scripts/eval/eval_best_adaptive.py --folder raw_telemetry_best
    python scripts/eval/eval_best_adaptive.py --folder raw_telemetry_no_judge

Usage:
    python scripts/run_judge_ablation.py
    python scripts/run_judge_ablation.py --source raw_telemetry_best --top 30
    python scripts/run_judge_ablation.py --out-dir raw_telemetry_no_judge --top 30
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from orchestrator.engine import OrchestratorEngine

ENGINE_TELEMETRY_FILE = ROOT / "experiment_telemetry.json"


# ---------------------------------------------------------------------------
# Query selection: pick top-N adaptive runs from source folder by node quality
# ---------------------------------------------------------------------------

def _selection_score(d: dict) -> float:
    et = d.get("engine_telemetry", {})
    nodes = et.get("got_dag_nodes", [])
    qualities = [n["quality_score"] for n in nodes if n.get("quality_score") is not None]
    if qualities:
        return sum(qualities) / len(qualities)
    fj = et.get("final_judge_score")
    return float(fj) / 10.0 if isinstance(fj, (int, float)) else 0.0


def load_top_queries(source_folder: Path, top_n: int) -> list[dict]:
    """
    Return top_n adaptive runs from source_folder sorted by node quality,
    each as {query_id, query} dicts. Duplicate queries (same text) are
    de-duplicated — keep the one with the highest score.
    """
    candidates = []
    for f in sorted(source_folder.glob("*.json")):
        try:
            d = json.loads(f.read_text())
            if d.get("ablation_mode") != "adaptive":
                continue
            et = d.get("engine_telemetry", {})
            query_text = et.get("user_query", d.get("query", "")).strip()
            if not query_text:
                continue
            candidates.append({
                "query_id": d.get("query_id", "?"),
                "query":    query_text,
                "score":    _selection_score(d),
            })
        except Exception:
            pass

    # De-duplicate by query text, keep highest score
    seen: dict[str, dict] = {}
    for c in candidates:
        key = c["query"][:120]  # first 120 chars as dedup key
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = c

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Telemetry helpers (mirrors run_ablation_benchmark.py)
# ---------------------------------------------------------------------------

def read_engine_telemetry() -> dict:
    try:
        with open(ENGINE_TELEMETRY_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[-1] if data else {}
        if isinstance(data, dict):
            return data
        return {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_run(telemetry: dict, metadata: dict, out_dir: Path) -> str:
    run_id = str(uuid.uuid4())[:8]
    record = {
        "run_id":    run_id,
        "timestamp": datetime.utcnow().isoformat(),
        **metadata,
        "engine_telemetry": telemetry,
    }
    out_path = out_dir / f"run_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_ablation(source_folder: Path, out_dir: Path, top_n: int) -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           JUDGE ABLATION BENCHMARK (no judges)           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Source : {source_folder}")
    print(f"  Output : {out_dir}")
    print(f"  Top-N  : {top_n}")
    print(f"  Mode   : adaptive  |  agent_judge=OFF  |  final_judge=OFF\n")

    # Disable both judges for this run
    cfg.Config.USE_AGENT_JUDGE = False
    cfg.Config.USE_FINAL_JUDGE = False

    queries = load_top_queries(source_folder, top_n)
    if not queries:
        print(f"  No adaptive queries found in {source_folder}")
        sys.exit(1)

    print(f"  Loaded {len(queries)} queries (score range: "
          f"{queries[-1]['score']:.3f} – {queries[0]['score']:.3f})\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    engine = OrchestratorEngine()

    for idx, item in enumerate(queries, 1):
        query    = item["query"]
        query_id = item["query_id"]
        print(f"\n══ Query {idx}/{len(queries)} [{query_id}]  sel={item['score']:.3f}")
        print(f"   {query[:100]}{'...' if len(query) > 100 else ''}")

        # Decompose once (same plan for consistency)
        cached_tasks, cached_decomp_name = None, None
        try:
            print("  🔧 Decomposing...")
            cached_tasks, cached_decomp_name = await engine.decompose_only(query)
            print(f"  🔧 Plan locked: {len(cached_tasks)} tasks via {cached_decomp_name.upper()}")
            await asyncio.sleep(3)
        except Exception as exc:
            print(f"  ✘ Decompose failed: {exc} — falling back to on-the-fly decomposition")

        meta = {
            "protocol":      "judge_ablation",
            "query_id":      query_id,
            "ablation_mode": "adaptive",
            "judge_ablation": {
                "agent_judge": False,
                "final_judge": False,
            },
            "query":         query,
            "dag_locked":    cached_tasks is not None,
        }

        try:
            if ENGINE_TELEMETRY_FILE.exists():
                os.remove(ENGINE_TELEMETRY_FILE)

            start_time = time.time()
            await engine.run(
                query,
                ablation_mode="adaptive",
                start_time=start_time,
                cached_tasks=cached_tasks,
                cached_decomposer_name=cached_decomp_name,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            telemetry = read_engine_telemetry()
            telemetry["observed_latency_ms"] = latency_ms
            telemetry["dag_locked"] = cached_tasks is not None

            run_id = save_run(telemetry, {**meta, "wall_latency_ms": latency_ms}, out_dir)
            print(f"  ✔ Saved: run_{run_id}.json  ({latency_ms}ms)")
            await asyncio.sleep(1)

        except Exception as exc:
            print(f"  ✘ Error: {exc}")
            save_run({}, {**meta, "error": str(exc), "failed": True}, out_dir)
            await asyncio.sleep(5)

    print(f"\n✅ Judge Ablation Complete. {len(queries)} runs saved to {out_dir.resolve()}")
    print(f"\nTo evaluate and compare:")
    print(f"  python scripts/eval/eval_best_adaptive.py --folder {source_folder.name}")
    print(f"  python scripts/eval/eval_best_adaptive.py --folder {out_dir.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  type=Path, default=ROOT / "raw_telemetry_best",
                        help="Folder to pull top-N adaptive queries from (default: raw_telemetry_best)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "raw_telemetry_no_judge",
                        help="Output folder for no-judge runs (default: raw_telemetry_no_judge)")
    parser.add_argument("--top",     type=int, default=30,
                        help="Number of queries to run (default: 30)")
    args = parser.parse_args()

    asyncio.run(run_ablation(args.source, args.out_dir, args.top))
