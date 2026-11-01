"""
rerun_diagnostician_cases.py
Re-runs only the queries that failed due to the dag_nodes_dump bug.

Detection logic:
  A run was affected if ANY of these are true:
    1. "failed": true  AND "error" contains "dag_nodes_dump"
    2. "failed": true  AND engine_telemetry is empty {}
    3. global_retries > 0 AND mode == "adaptive"  (original check, kept as fallback)
"""

import asyncio, json, sys, time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from orchestrator.engine import OrchestratorEngine

RAW_TELEMETRY_DIR = _PROJECT_ROOT / "raw_telemetry"


def find_failed_runs() -> list[dict]:
    if not RAW_TELEMETRY_DIR.exists():
        print(f"❌ Directory not found: {RAW_TELEMETRY_DIR}")
        return []

    failed    = []
    all_files = sorted(RAW_TELEMETRY_DIR.glob("run_*.json"))
    print(f"Scanning {len(all_files)} telemetry file(s) in {RAW_TELEMETRY_DIR.name}/...\n")

    for path in all_files:
        with open(path) as f:
            try:
                run = json.load(f)
            except json.JSONDecodeError:
                print(f"  [warn] Could not parse {path.name} — skipping")
                continue

        t       = run.get("engine_telemetry", {})
        mode    = run.get("ablation_mode", t.get("ablation_mode", "unknown"))
        score   = t.get("final_judge_score", run.get("score", "?"))
        query   = run.get("query", t.get("user_query", ""))
        error   = run.get("error", "")
        is_failed = run.get("failed", False)
        retries = t.get("global_retries", 0)
        tele_empty = (t == {} or t is None)

        # ── Detection ─────────────────────────────────────────────────────────
        reason = None

        if is_failed and "dag_nodes_dump" in str(error):
            reason = f"crashed with dag_nodes_dump error"

        elif is_failed and tele_empty:
            reason = f"failed with empty telemetry (likely same bug)"

        elif retries > 0 and mode == "adaptive":
            reason = f"diagnostician triggered (global_retries={retries})"

        # ── Report ────────────────────────────────────────────────────────────
        if reason:
            failed.append({
                "file":   path.name,
                "query":  query,
                "mode":   mode,
                "score":  score,
                "reason": reason,
            })
            print(f"  ✘ NEEDS RERUN | {path.name} | mode={mode} | {reason}")
            print(f"                  query: {query[:70]}...")
        else:
            print(f"  ✔ clean       | {path.name} | mode={mode} score={score}")

    print(f"\n→ {len(failed)} run(s) need re-running out of {len(all_files)} total.\n")
    return failed


async def main():
    failed_runs = find_failed_runs()

    if not failed_runs:
        print("✔ No runs need re-running. All results are clean.")
        return

    print("Starting re-runs...\n")
    engine = OrchestratorEngine()

    for i, run in enumerate(failed_runs):
        query = run["query"]
        if not query:
            print(f"  [skip] {run['file']} — no query stored, cannot re-run")
            continue

        print(f"\n── Re-running {i+1}/{len(failed_runs)}")
        print(f"   File   : {run['file']}")
        print(f"   Reason : {run['reason']}")
        print(f"   Mode   : {run['mode']}")
        print(f"   Query  : {query[:80]}...")

        try:
            start = time.time()
            await engine.run(
                query,
                ablation_mode=run["mode"],
                start_time=start,
            )
            print(f"   ✔ Done")
            await asyncio.sleep(35)
        except Exception as e:
            print(f"   ✘ Error: {e}")
            await asyncio.sleep(90)

    print("\n✅ Re-runs complete.")
    print("   New runs saved to raw_telemetry/ as new run_*.json files.")
    print("   Now refresh your metrics and graphs:")
    print("     python scripts/compute_metrics.py")
    print("     python scripts/visualize_metrics.py")


if __name__ == "__main__":
    asyncio.run(main())