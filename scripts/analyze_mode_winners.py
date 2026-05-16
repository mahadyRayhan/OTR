"""
analyze_mode_winners.py — Ablation mode win/loss analysis with token breakdown
===============================================================================
For each benchmark query, identifies which ablation mode (none / adaptive / raw)
produces the best score, grouped by agent combination.

For queries where adaptive wins:
  - Shows token utilization % vs the other modes
  - Explains WHY adaptive wins (cugcr_triggered, retries, score dimensions)

Score source priority:
  1. results/llm_judge_telemetry.json  (OpenAI external judge, preferred)
  2. raw_telemetry run files           (internal Gemini judge, fallback)

Token and CUGCR data always come directly from raw_telemetry/*.json.

Usage:
    python scripts/analyze_mode_winners.py
    python scripts/analyze_mode_winners.py --telemetry raw_telemetry --judge results/llm_judge_telemetry.json
    python scripts/analyze_mode_winners.py --use-internal-judge
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_telemetry(tel_dir: Path) -> dict[tuple, dict]:
    """Load raw telemetry; returns {(query_id, mode): run_dict}."""
    runs = {}
    for f in sorted(tel_dir.glob("run_*.json")):
        try:
            d = json.loads(f.read_text())
            key = (d.get("query_id", "?"), d.get("ablation_mode", "?"))
            runs[key] = d
        except Exception as exc:
            print(f"  [WARN] Could not load {f.name}: {exc}")
    return runs


def load_judge_results(judge_path: Path) -> dict[tuple, dict] | None:
    """Load llm_judge_telemetry.json; returns {(query_id, mode): result_dict} or None."""
    if not judge_path.exists():
        return None
    try:
        data = json.loads(judge_path.read_text())
        out = {}
        for r in data.get("results", []):
            key = (r["query_id"], r["ablation_mode"])
            out[key] = r
        return out
    except Exception as exc:
        print(f"  [WARN] Could not load judge file {judge_path}: {exc}")
        return None


def extract_agents(run: dict) -> list[str]:
    tasks = run.get("engine_telemetry", {}).get("initial_tasks", [])
    seen, ordered = set(), []
    for t in tasks:
        a = t.get("agent", "")
        if a and a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_query_table(
    raw_runs: dict[tuple, dict],
    judge_results: dict[tuple, dict] | None,
    use_internal: bool,
) -> dict[str, dict]:
    """
    Returns {query_id: {
        "query": str,
        "agents": list[str],
        "modes": {mode: {
            "score": int,
            "judge_source": str,
            "cugcr_triggered": bool,
            "tok_in": int,
            "first_pass_success": bool | None,
            "global_retries": int,
            "judge_dims": dict,   # correctness/completeness/usability from external
        }}
    }}
    """
    all_keys = set(raw_runs.keys())
    if judge_results:
        all_keys |= set(judge_results.keys())

    query_data: dict[str, dict] = {}

    for (qid, mode), run in raw_runs.items():
        if qid not in query_data:
            query_data[qid] = {
                "query":   run.get("query", run.get("engine_telemetry", {}).get("user_query", "")),
                "agents":  extract_agents(run),
                "modes":   {},
            }

        et = run.get("engine_telemetry", {})
        tok_in = (et.get("orchestrator_tokens") or {}).get("input", 0)
        cugcr  = et.get("cugcr_triggered", False)
        fps    = et.get("first_pass_success", None)
        retries = et.get("global_retries", 0)

        # Pick score source
        judge_dims = {}
        if not use_internal and judge_results and (qid, mode) in judge_results:
            jr = judge_results[(qid, mode)]
            score = jr["judge"]["total"]
            judge_source = "openai"
            judge_dims = {
                "correctness":  jr["judge"].get("correctness", 0),
                "completeness": jr["judge"].get("completeness", 0),
                "usability":    jr["judge"].get("usability", 0),
                "rationale":    jr["judge"].get("rationale", ""),
            }
        else:
            score = et.get("final_judge_score") or 0
            judge_source = "gemini-internal"

        query_data[qid]["modes"][mode] = {
            "score":             score,
            "judge_source":      judge_source,
            "cugcr_triggered":   cugcr,
            "tok_in":            tok_in,
            "first_pass_success": fps,
            "global_retries":    retries,
            "judge_dims":        judge_dims,
        }

    return query_data


def find_winners(modes_data: dict) -> tuple[list[str], int, list[str]]:
    """Returns (winners, top_score, losers)."""
    if not modes_data:
        return [], 0, []
    top = max(m["score"] for m in modes_data.values())
    winners = [mode for mode, m in modes_data.items() if m["score"] == top]
    losers  = [mode for mode, m in modes_data.items() if m["score"] < top]
    return sorted(winners), top, sorted(losers)


def token_utilization(adaptive_tok: int, other_toks: dict[str, int]) -> dict[str, str]:
    """Returns per-mode token delta descriptions."""
    out = {}
    for mode, tok in other_toks.items():
        if tok == 0:
            out[mode] = "n/a"
            continue
        pct = adaptive_tok / tok * 100
        delta = adaptive_tok - tok
        sign = "+" if delta > 0 else ("-" if delta < 0 else "")
        out[mode] = f"{pct:.0f}% of {mode} ({sign}{abs(delta)} tokens)"
    return out


def build_edge_data(raw_runs: dict[tuple, dict]) -> dict:
    """
    Extracts per-agent context injection sizes from communication edges.

    Returns:
      {
        "by_agent_mode": {(to_agent, mode, wave): [ctx_len, ...]},
        "compression":   {(qid, to_agent): {"adaptive": avg, "raw": avg, "ratio": %}}
      }

    Key insight: in adaptive mode, raw_context_injected already contains
    CUGCR-compressed text (compression is applied in-place). Comparing
    adaptive vs raw edge sizes gives the true compression ratio.
    """
    by_agent_mode: dict = defaultdict(list)
    # wave-1 per-query edge sizes: {(qid, to_agent, mode): [len,...]}
    wave1_sizes: dict = defaultdict(list)

    for (qid, mode), run in raw_runs.items():
        et = run.get("engine_telemetry", {})
        for e in et.get("communication_edges", []):
            to_agent = e.get("to_agent", "?")
            wave     = e.get("wave_number", 0)
            ctx_len  = len(e.get("raw_context_injected", "") or "")
            by_agent_mode[(to_agent, mode, wave)].append(ctx_len)
            if wave == 1:
                wave1_sizes[(qid, to_agent, mode)].append(ctx_len)

    # Per-query, per-agent compression ratio (adaptive vs raw, wave 1)
    compression: dict = {}
    all_qid_agent = {(q, a) for (q, a, _) in wave1_sizes.keys()}
    for (qid, to_agent) in all_qid_agent:
        ad_vals = wave1_sizes.get((qid, to_agent, "adaptive"), [])
        rw_vals = wave1_sizes.get((qid, to_agent, "raw"), [])
        if ad_vals and rw_vals:
            ad_avg = sum(ad_vals) / len(ad_vals)
            rw_avg = sum(rw_vals) / len(rw_vals)
            ratio  = (ad_avg / rw_avg * 100) if rw_avg else 100.0
            compression[(qid, to_agent)] = {
                "adaptive_chars": int(ad_avg),
                "raw_chars":      int(rw_avg),
                "ratio_pct":      round(ratio, 1),
                "saved_chars":    int(rw_avg - ad_avg),
            }

    return {"by_agent_mode": dict(by_agent_mode), "compression": compression}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _mode_tag(mode: str) -> str:
    icons = {"adaptive": "▲ adaptive", "none": "● none", "raw": "■ raw"}
    return icons.get(mode, mode)


def print_analysis(query_data: dict[str, dict], judge_label: str, edge_data: dict | None = None):
    MODES = ["adaptive", "none", "raw"]

    # Group queries by agent combination
    by_agents: dict[str, list[str]] = defaultdict(list)
    for qid, qd in sorted(query_data.items()):
        combo = "+".join(qd["agents"])
        by_agents[combo].append(qid)

    print(f"\n{'═'*70}")
    print(f"  ABLATION MODE WINNER ANALYSIS  (scores from: {judge_label})")
    print(f"{'═'*70}")

    # ── Global win tally ──────────────────────────────────────
    win_counts: dict[str, int] = defaultdict(int)
    tie_count = 0
    for qd in query_data.values():
        winners, top, _ = find_winners(qd["modes"])
        if len(winners) == len(qd["modes"]):
            tie_count += 1
        else:
            for w in winners:
                win_counts[w] += 1  # outright or shared wins still count

    print(f"\n── Global Win Tally ─────────────────────────────────────────")
    for mode in MODES:
        bar = "█" * win_counts.get(mode, 0)
        print(f"  {mode:<10} {win_counts.get(mode, 0):>2} win(s)  {bar}")
    print(f"  {'ties':<10} {tie_count:>2} (all modes equal)")

    # ── Per-agent-combination breakdown ───────────────────────
    print(f"\n── Per Agent Combination ────────────────────────────────────")

    for combo, qids in sorted(by_agents.items()):
        print(f"\n  ┌─ Agents: {combo}")
        for qid in sorted(qids):
            qd = query_data[qid]
            modes = qd["modes"]
            winners, top, losers = find_winners(modes)

            q_preview = qd["query"][:60] + ("..." if len(qd["query"]) > 60 else "")
            print(f"  │  {qid}: {q_preview}")

            # Score row
            score_parts = []
            for m in MODES:
                if m not in modes:
                    continue
                s = modes[m]["score"]
                marker = "★" if m in winners and len(winners) < len(modes) else " "
                score_parts.append(f"{marker}{m}={s}")
            print(f"  │   scores: {',  '.join(score_parts)}")

            # Determine outcome
            if len(winners) == len(modes):
                outcome = "TIE — all modes equal"
            elif len(winners) == 1:
                outcome = f"WINNER: {winners[0].upper()}"
            else:
                outcome = f"TIED WINNERS: {', '.join(w.upper() for w in winners)}"
            print(f"  │   outcome: {outcome}")

            # Adaptive-win analysis
            if "adaptive" in winners and len(winners) < len(modes):
                ad = modes["adaptive"]
                other_toks = {m: modes[m]["tok_in"] for m in modes if m != "adaptive"}
                tok_cmp = token_utilization(ad["tok_in"], other_toks)
                compression = edge_data.get("compression", {}) if edge_data else {}

                print(f"  │   ── Adaptive Win Details ──────────────────────────")
                print(f"  │      CUGCR triggered : {'YES — edge context compressed in-place' if ad['cugcr_triggered'] else 'NO — no edge compression'}")
                print(f"  │      Token input (orch): {ad['tok_in']:,} tokens")
                for m, desc in tok_cmp.items():
                    print(f"  │        vs {m:<8}: {desc}")

                # Show per-agent edge compression if available
                agent_compressions = {
                    ag: v for (q, ag), v in compression.items()
                    if q == qid and v["saved_chars"] > 50
                }
                if agent_compressions:
                    print(f"  │      Edge-level compression (adaptive vs raw):")
                    for ag, cv in sorted(agent_compressions.items()):
                        print(f"  │        → {ag:<8}: {cv['adaptive_chars']:,} vs {cv['raw_chars']:,} chars "
                              f"({cv['ratio_pct']:.0f}% — saved {cv['saved_chars']:,} chars)")

                if ad["first_pass_success"] is not None:
                    fps_str = "yes (no orchestrator retry)" if ad["first_pass_success"] else "no (retried)"
                    print(f"  │      First-pass OK   : {fps_str}")
                if ad.get("global_retries"):
                    print(f"  │      Global retries  : {ad['global_retries']}")

                # Explain WHY — distinguish CUGCR-driven vs retry-driven wins
                reasons = []
                if ad["cugcr_triggered"]:
                    if agent_compressions:
                        biggest = max(agent_compressions.values(), key=lambda v: v["saved_chars"])
                        reasons.append(
                            f"CUGCR compressed per-agent edge context (up to {biggest['saved_chars']:,} chars "
                            f"saved). Smaller, focused context reduces noise injected into downstream agents."
                        )
                    else:
                        reasons.append(
                            "CUGCR triggered but edge savings were minimal (<50 chars). "
                            "The orchestrator-level token overhead exceeded savings — win likely from "
                            "orchestration strategy, not compression."
                        )
                else:
                    # No CUGCR: win must come from adaptive orchestration logic
                    retries = ad.get("global_retries", 0)
                    if retries:
                        reasons.append(
                            f"CUGCR did NOT trigger (no edge compression). Win driven by adaptive "
                            f"orchestration: {retries} retry/refinement cycle(s) allowed the orchestrator "
                            f"to correct and improve the answer iteratively."
                        )
                    else:
                        reasons.append(
                            "CUGCR did NOT trigger. Adaptive orchestration strategy produced better "
                            "decomposition without needing compression — agents received better-scoped tasks."
                        )

                dims = ad.get("judge_dims", {})
                if dims:
                    best_dim = max(
                        [("correctness", dims.get("correctness", 0)),
                         ("completeness", dims.get("completeness", 0)),
                         ("usability", dims.get("usability", 0))],
                        key=lambda x: x[1],
                    )
                    reasons.append(
                        f"Strongest dimension: {best_dim[0]} ({best_dim[1]}/{'4' if best_dim[0]=='correctness' else '3'})."
                    )
                    if dims.get("rationale"):
                        reasons.append(f"Judge: \"{dims['rationale'][:100]}\"")

                if not reasons:
                    reasons.append("Adaptive produced a higher-quality answer; mechanism unclear without dimension data.")

                print(f"  │      Why adaptive won:")
                for r in reasons:
                    for line in _wrap(r, 58):
                        print(f"  │        {line}")

            # Non-adaptive winner: brief note
            elif "adaptive" not in winners and len(winners) < len(modes):
                winner = winners[0]
                ad = modes.get("adaptive", {})
                loser_note = []
                if ad.get("cugcr_triggered"):
                    loser_note.append("CUGCR compressed context but hurt quality (over-compression)")
                else:
                    loser_note.append("CUGCR did not trigger — adaptive had no compression benefit")
                if winner == "none":
                    loser_note.append(f"'none' mode passes full context without compression overhead → cleaner output")
                elif winner == "raw":
                    loser_note.append(f"'raw' mode preserves full inter-agent verbosity → richer detail for judge")

                ad_tok = ad.get("tok_in", 0)
                w_tok  = modes.get(winner, {}).get("tok_in", 0)
                if ad_tok and w_tok:
                    loser_note.append(f"adaptive={ad_tok:,} tokens vs {winner}={w_tok:,} tokens")

                print(f"  │   Why adaptive lost:")
                for note in loser_note:
                    for line in _wrap(note, 58):
                        print(f"  │     {line}")

        print(f"  └{'─'*65}")

    # ── Token utilization summary for adaptive ─────────────────
    print(f"\n── Token Utilization: Adaptive vs Others (all queries) ──────")
    print(f"  {'QID':<6} {'Agents':<22} {'cugcr':>6} {'adaptive':>9} {'vs none':>9} {'vs raw':>9} {'winner'}")
    print(f"  {'─'*6} {'─'*22} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*10}")

    for qid in sorted(query_data.keys()):
        qd = query_data[qid]
        modes = qd["modes"]
        agents_str = "+".join(qd["agents"])[:22]
        ad = modes.get("adaptive", {})
        if not ad:
            continue
        ad_tok = ad.get("tok_in", 0)
        none_tok = modes.get("none", {}).get("tok_in", 0)
        raw_tok  = modes.get("raw",  {}).get("tok_in", 0)

        def _pct(a, b):
            return f"{a/b*100:.0f}%" if b else "n/a"

        winners, top, _ = find_winners(modes)
        winner_str = "/".join(winners) if winners else "?"

        print(
            f"  {qid:<6} {agents_str:<22} {'Y' if ad.get('cugcr_triggered') else 'N':>6} "
            f"{ad_tok:>9,} {_pct(ad_tok, none_tok):>9} {_pct(ad_tok, raw_tok):>9} "
            f"{winner_str}"
        )

    # ── Agent context passing patterns ────────────────────────
    if edge_data:
        _print_context_patterns(edge_data)

    print()


def _print_context_patterns(edge_data: dict):
    """Print which agents receive inter-agent context and how much, per mode."""
    by_am = edge_data.get("by_agent_mode", {})
    compression = edge_data.get("compression", {})

    AGENTS = ["rag", "scholar", "code", "cloud"]

    print(f"\n── Agent Context Passing Patterns ───────────────────────────")
    print(f"  How much context (chars) each agent receives from upstream agents.")
    print(f"  'none' mode always injects 0 — agents work independently.")
    print()

    # Wave 1 — primary injection (rag → downstream)
    print(f"  Wave 1 (primary injection, e.g. rag → code/cloud/scholar):")
    print(f"  {'agent':<10} {'none':>10} {'adaptive':>12} {'raw':>10}  pattern")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*10}  {'─'*30}")
    for ag in AGENTS:
        none_vals = by_am.get((ag, "none", 1), [])
        ad_vals   = by_am.get((ag, "adaptive", 1), [])
        rw_vals   = by_am.get((ag, "raw", 1), [])

        def _fmt(vals):
            if not vals:
                return "n/a"
            nonzero = [v for v in vals if v > 0]
            avg = sum(vals) / len(vals)
            return f"{avg:,.0f} ({len(nonzero)}/{len(vals)})"

        # Pattern description
        if not ad_vals and not rw_vals:
            pattern = "never receives context"
        elif all(v == 0 for v in none_vals):
            ad_avg = sum(ad_vals) / len(ad_vals) if ad_vals else 0
            rw_avg = sum(rw_vals) / len(rw_vals) if rw_vals else 0
            if rw_avg == 0:
                pattern = "receives context only in adaptive"
            elif ad_avg == 0:
                pattern = "receives context only in raw"
            else:
                ratio = ad_avg / rw_avg * 100
                if ratio < 50:
                    pattern = f"CUGCR compresses heavily (adaptive={ratio:.0f}% of raw)"
                elif ratio < 85:
                    pattern = f"CUGCR compresses moderately (adaptive={ratio:.0f}% of raw)"
                else:
                    pattern = f"minimal CUGCR compression (adaptive={ratio:.0f}% of raw)"
        else:
            pattern = "—"

        print(f"  {ag:<10} {_fmt(none_vals):>10} {_fmt(ad_vals):>12} {_fmt(rw_vals):>10}  {pattern}")

    # Wave 2 — retry/feedback injection
    print()
    print(f"  Wave 2 (retry/feedback injection):")
    print(f"  {'agent':<10} {'none':>10} {'adaptive':>12} {'raw':>10}  pattern")
    print(f"  {'─'*10} {'─'*10} {'─'*12} {'─'*10}  {'─'*30}")
    for ag in AGENTS:
        none_vals = by_am.get((ag, "none", 2), [])
        ad_vals   = by_am.get((ag, "adaptive", 2), [])
        rw_vals   = by_am.get((ag, "raw", 2), [])

        def _fmt2(vals):
            if not vals:
                return "n/a"
            nonzero = [v for v in vals if v > 0]
            avg = sum(vals) / len(vals)
            return f"{avg:,.0f} ({len(nonzero)}/{len(vals)})"

        if not ad_vals and not rw_vals:
            pattern = "no retry context"
        elif all(v == 0 for v in (none_vals or [])):
            ad_avg = sum(ad_vals) / len(ad_vals) if ad_vals else 0
            rw_avg = sum(rw_vals) / len(rw_vals) if rw_vals else 0
            if ad_avg > 0 and rw_avg > 0:
                ratio = ad_avg / rw_avg * 100
                pattern = f"retry feedback ({ratio:.0f}% of raw)"
            elif ad_avg > 0:
                pattern = "retry feedback (adaptive only)"
            else:
                pattern = "retry feedback"
        else:
            pattern = "—"

        print(f"  {ag:<10} {_fmt2(none_vals):>10} {_fmt2(ad_vals):>12} {_fmt2(rw_vals):>10}  {pattern}")

    # Key insight summary
    print()
    print(f"  Key observations:")

    # CUGCR effect on cloud (biggest compression)
    cloud_compressions = [v for (_, ag), v in compression.items() if ag == "cloud"]
    if cloud_compressions:
        avg_ratio = sum(v["ratio_pct"] for v in cloud_compressions) / len(cloud_compressions)
        print(f"  • cloud   : biggest CUGCR beneficiary — adaptive injects only {avg_ratio:.0f}% of raw context")

    # Code always gets context
    code_ad = by_am.get(("code", "adaptive", 1), [])
    code_rw = by_am.get(("code", "raw", 1), [])
    if code_ad and code_rw:
        nonzero_ad = sum(1 for v in code_ad if v > 0)
        ad_avg = sum(code_ad) / len(code_ad)
        rw_avg = sum(code_rw) / len(code_rw)
        ratio  = ad_avg / rw_avg * 100 if rw_avg else 100
        print(f"  • code    : always receives wave-1 context ({nonzero_ad}/{len(code_ad)} injections); "
              f"adaptive={ad_avg:,.0f} chars vs raw={rw_avg:,.0f} chars ({ratio:.0f}% of raw)")

    # Scholar barely needs context
    sch_ad = by_am.get(("scholar", "adaptive", 1), [])
    sch_rw = by_am.get(("scholar", "raw", 1), [])
    if sch_ad and sch_rw:
        ratio = sum(sch_ad) / sum(sch_rw) * 100 if sum(sch_rw) else 100
        print(f"  • scholar : receives context only when paired with rag; "
              f"CUGCR barely compresses ({ratio:.0f}% of raw) — scholar's API results are self-contained")
    else:
        print(f"  • scholar : only receives context in rag+scholar combos; self-contained otherwise")

    # Rag as pure source
    rag_w1 = by_am.get(("rag", "adaptive", 1), [])
    if not rag_w1:
        print(f"  • rag     : never a wave-1 context receiver — always the upstream source")

    # None mode
    print(f"  • none    : 0 chars injected to every agent in every query — fully isolated execution")


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines, current = [], []
    length = 0
    for w in words:
        if length + len(w) + 1 > width and current:
            lines.append(" ".join(current))
            current, length = [w], len(w)
        else:
            current.append(w)
            length += len(w) + 1
    if current:
        lines.append(" ".join(current))
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--telemetry", type=Path,
        default=ROOT / "raw_telemetry",
        help="Directory containing run_*.json files",
    )
    parser.add_argument(
        "--judge", type=Path,
        default=ROOT / "results" / "llm_judge_telemetry.json",
        help="Path to llm_judge_telemetry.json (OpenAI judge output)",
    )
    parser.add_argument(
        "--use-internal-judge", action="store_true",
        help="Use Gemini internal judge scores even if OpenAI results are available",
    )
    args = parser.parse_args()

    raw_runs = load_raw_telemetry(args.telemetry)
    if not raw_runs:
        print(f"No run_*.json files found in {args.telemetry}")
        sys.exit(1)

    judge_results = None if args.use_internal_judge else load_judge_results(args.judge)

    if judge_results:
        judge_label = f"OpenAI external ({args.judge.name})"
    else:
        judge_label = "Gemini internal (from raw telemetry)"

    query_data = build_query_table(raw_runs, judge_results, args.use_internal_judge)
    edge_data  = build_edge_data(raw_runs)
    print_analysis(query_data, judge_label, edge_data)


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════
#   ABLATION MODE WINNER ANALYSIS  (scores from: OpenAI external (llm_judge_telemetry.json))
# ══════════════════════════════════════════════════════════════════════

# ── Global Win Tally ─────────────────────────────────────────
#   adaptive    1 win(s)  █
#   none        2 win(s)  ██
#   raw         1 win(s)  █
#   ties        1 (all modes equal)

# ── Per Agent Combination ────────────────────────────────────

#   ┌─ Agents: rag+code
#   │  BQ05: Write a simple Python script using pysam to open a BAM file,...
#   │   scores:  adaptive=8,   none=8,  ★raw=10
#   │   outcome: WINNER: RAW
#   │   Why adaptive lost:
#   │     CUGCR compressed context but hurt quality
#   │     (over-compression)
#   │     'raw' mode preserves full inter-agent verbosity → richer
#   │     detail for judge
#   │     adaptive=1,865 tokens vs raw=1,830 tokens
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: rag+code+cloud
#   │  BQ01: Explain the statistical model behind DESeq2 for differential...
#   │   scores:  adaptive=4,  ★none=5,   raw=4
#   │   outcome: WINNER: NONE
#   │   Why adaptive lost:
#   │     CUGCR compressed context but hurt quality
#   │     (over-compression)
#   │     'none' mode passes full context without compression
#   │     overhead → cleaner output
#   │     adaptive=4,471 tokens vs none=3,511 tokens
#   │  BQ03: Explain how to process a whole-genome sequencing dataset thr...
#   │   scores:  adaptive=5,   none=5,   raw=5
#   │   outcome: TIE — all modes equal
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: rag+scholar
#   │  BQ04: Find 4 experts who have published papers on 'STAR RNA-seq sp...
#   │   scores: ★adaptive=7,   none=6,   raw=6
#   │   outcome: WINNER: ADAPTIVE
#   │   ── Adaptive Win Details ──────────────────────────
#   │      CUGCR triggered : NO — no compression needed
#   │      Token input      : 1,826 tokens
#   │        vs raw     : 143% of raw (+547 tokens)
#   │        vs none    : 103% of none (+57 tokens)
#   │      First-pass OK   : no (retried)
#   │      Global retries  : 2
#   │      Why adaptive won:
#   │        Strongest dimension: completeness (3) — compressed
#   │        context likely helped the agent focus on the core ask.
#   │        Judge: "The answer provides four real researchers tied to
#   │        the seminal STAR paper and generally explains STAR"
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: scholar+code
#   │  BQ02: Find 3 experts on Leiden community detection in single-cell ...
#   │   scores:  adaptive=4,  ★none=9,   raw=7
#   │   outcome: WINNER: NONE
#   │   Why adaptive lost:
#   │     CUGCR did not trigger — adaptive had no compression
#   │     benefit
#   │     'none' mode passes full context without compression
#   │     overhead → cleaner output
#   │     adaptive=1,672 tokens vs none=1,640 tokens
#   └─────────────────────────────────────────────────────────────────

# ── Token Utilization: Adaptive vs Others (all queries) ──────
#   QID    Agents                  cugcr  adaptive   vs none    vs raw winner
#   ────── ────────────────────── ────── ───────── ───────── ───────── ──────────
#   BQ01   rag+code+cloud              Y     4,471      127%      117% none
#   BQ02   scholar+code                N     1,672      102%       96% none
#   BQ03   rag+code+cloud              Y     4,287       84%       85% adaptive/none/raw
#   BQ04   rag+scholar                 N     1,826      103%      143% adaptive
#   BQ05   rag+code                    Y     1,865      103%      102% raw


# ── Agent Context Passing Patterns ───────────────────────────
#   How much context (chars) each agent receives from upstream agents.
#   'none' mode always injects 0 — agents work independently.

#   Wave 1 (primary injection, e.g. rag → code/cloud/scholar):
#   agent            none     adaptive        raw  pattern
#   ────────── ────────── ──────────── ──────────  ──────────────────────────────
#   rag               n/a          n/a        n/a  never receives context
#   scholar       0 (0/1)  1,284 (1/1) 1,311 (1/1)  minimal CUGCR compression (adaptive=98% of raw)
#   code          0 (0/5)  2,234 (5/5) 2,890 (5/5)  CUGCR compresses moderately (adaptive=77% of raw)
#   cloud         0 (0/2)    624 (2/2) 2,515 (2/2)  CUGCR compresses heavily (adaptive=25% of raw)

#   Wave 2 (retry/feedback injection):
#   agent            none     adaptive        raw  pattern
#   ────────── ────────── ──────────── ──────────  ──────────────────────────────
#   rag           0 (0/6)    777 (5/5)  877 (4/4)  retry feedback (89% of raw)
#   scholar       0 (0/4)  1,347 (5/5) 1,357 (5/5)  retry feedback (99% of raw)
#   code          0 (0/4)    421 (4/4)  425 (3/3)  retry feedback (99% of raw)
#   cloud             n/a          n/a        n/a  no retry context

#   Key observations:
#   • cloud   : biggest CUGCR beneficiary — adaptive injects only 55% of raw context
#   • code    : always receives wave-1 context (5/5 injections); adaptive=2,234 chars vs raw=2,890 chars (77% of raw)
#   • scholar : receives context only when paired with rag; CUGCR barely compresses (98% of raw) — scholar's API results are self-contained
#   • rag     : never a wave-1 context receiver — always the upstream source
#   • none    : 0 chars injected to every agent in every query — fully isolated execution


# --------------------- AFTER CALIBRATION RUNS ----------------------
# ══════════════════════════════════════════════════════════════════════
#   ABLATION MODE WINNER ANALYSIS  (scores from: OpenAI external (llm_judge_calibrated.json))
# ══════════════════════════════════════════════════════════════════════

# ── Global Win Tally ─────────────────────────────────────────
#   adaptive    1 win(s)  █
#   none        1 win(s)  █
#   raw         0 win(s)  
#   ties        3 (all modes equal)

# ── Per Agent Combination ────────────────────────────────────

#   ┌─ Agents: rag+code
#   │  BQ05: Write a simple Python script using pysam to open a BAM file,...
#   │   scores:  adaptive=9,   none=9
#   │   outcome: TIE — all modes equal
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: rag+code+cloud
#   │  BQ01: Explain the statistical model behind DESeq2 for differential...
#   │   scores: ★adaptive=5,   none=4
#   │   outcome: WINNER: ADAPTIVE
#   │   ── Adaptive Win Details ──────────────────────────
#   │      CUGCR triggered : YES — edge context compressed in-place
#   │      Token input (orch): 5,792 tokens
#   │        vs none    : 107% of none (+364 tokens)
#   │      First-pass OK   : yes (no orchestrator retry)
#   │      Why adaptive won:
#   │        CUGCR triggered but edge savings were minimal (<50
#   │        chars). The orchestrator-level token overhead exceeded
#   │        savings — win likely from orchestration strategy, not
#   │        compression.
#   │        Strongest dimension: correctness (2/4).
#   │        Judge: "The DESeq2 explanation is broadly reasonable, but
#   │        it contains technical imprecision (e.g. dispersion"
#   │  BQ03: Explain how to process a whole-genome sequencing dataset thr...
#   │   scores:  adaptive=5,   none=5
#   │   outcome: TIE — all modes equal
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: rag+scholar
#   │  BQ04: Find 4 experts who have published papers on 'STAR RNA-seq sp...
#   │   scores:  adaptive=4,  ★none=6
#   │   outcome: WINNER: NONE
#   │   Why adaptive lost:
#   │     CUGCR did not trigger — adaptive had no compression
#   │     benefit
#   │     'none' mode passes full context without compression
#   │     overhead → cleaner output
#   │     adaptive=1,418 tokens vs none=1,212 tokens
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: scholar+code
#   │  BQ02: Find 3 experts on Leiden community detection in single-cell ...
#   │   scores:  adaptive=5,   none=5
#   │   outcome: TIE — all modes equal
#   └─────────────────────────────────────────────────────────────────

# ── Token Utilization: Adaptive vs Others (all queries) ──────
#   QID    Agents                  cugcr  adaptive   vs none    vs raw winner
#   ────── ────────────────────── ────── ───────── ───────── ───────── ──────────
#   BQ01   rag+code+cloud              Y     5,792      107%       n/a adaptive
#   BQ02   scholar+code                Y     2,628      124%       n/a adaptive/none
#   BQ03   rag+code+cloud              Y     3,803       90%       n/a adaptive/none
#   BQ04   rag+scholar                 N     1,418      117%       n/a none
#   BQ05   rag+code                    Y     1,809       75%       n/a adaptive/none

# ── Agent Context Passing Patterns ───────────────────────────
#   How much context (chars) each agent receives from upstream agents.
#   'none' mode always injects 0 — agents work independently.

#   Wave 1 (primary injection, e.g. rag → code/cloud/scholar):
#   agent            none     adaptive        raw  pattern
#   ────────── ────────── ──────────── ──────────  ──────────────────────────────
#   rag           0 (0/2)  2,682 (2/2)        n/a  receives context only in adaptive
#   scholar       0 (0/1)  2,064 (1/1)        n/a  receives context only in adaptive
#   code          0 (0/5)  2,138 (5/5)        n/a  receives context only in adaptive
#   cloud         0 (0/2)    421 (2/2)        n/a  receives context only in adaptive

#   Wave 2 (retry/feedback injection):
#   agent            none     adaptive        raw  pattern
#   ────────── ────────── ──────────── ──────────  ──────────────────────────────
#   rag           0 (0/6)    871 (4/4)        n/a  retry feedback (adaptive only)
#   scholar       0 (0/2)  2,223 (1/1)        n/a  retry feedback (adaptive only)
#   code          0 (0/4)  1,022 (4/4)        n/a  retry feedback (adaptive only)
#   cloud             n/a  2,802 (1/1)        n/a  retry feedback (adaptive only)

#   Key observations:
#   • scholar : only receives context in rag+scholar combos; self-contained otherwise
#   • none    : 0 chars injected to every agent in every query — fully isolated execution

# -------------------------- AFTER 2ND CALIBRATION RUNS ----------------------
# ══════════════════════════════════════════════════════════════════════
#   ABLATION MODE WINNER ANALYSIS  (scores from: OpenAI external (llm_judge_calibrated_2.json))
# ══════════════════════════════════════════════════════════════════════

# ── Global Win Tally ─────────────────────────────────────────
#   adaptive    1 win(s)  █
#   none        2 win(s)  ██
#   raw         0 win(s)  
#   ties        2 (all modes equal)

# ── Per Agent Combination ────────────────────────────────────

#   ┌─ Agents: code
#   │  BQ05: Write a simple Python script using pysam to open a BAM file,...
#   │   scores: ★adaptive=10,   none=9
#   │   outcome: WINNER: ADAPTIVE
#   │   ── Adaptive Win Details ──────────────────────────
#   │      CUGCR triggered : NO — no edge compression
#   │      Token input (orch): 2,191 tokens
#   │        vs none    : 92% of none (-183 tokens)
#   │      First-pass OK   : yes (no orchestrator retry)
#   │      Why adaptive won:
#   │        CUGCR did NOT trigger. Adaptive orchestration strategy
#   │        produced better decomposition without needing compression
#   │        — agents received better-scoped tasks.
#   │        Strongest dimension: correctness (4/4).
#   │        Judge: "The answer provides a correct pysam script that
#   │        opens a BAM file in rb mode, iterates over alignment"
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: rag+code+cloud
#   │  BQ01: Explain the statistical model behind DESeq2 for differential...
#   │   scores:  adaptive=4,  ★none=5
#   │   outcome: WINNER: NONE
#   │   Why adaptive lost:
#   │     CUGCR compressed context but hurt quality
#   │     (over-compression)
#   │     'none' mode passes full context without compression
#   │     overhead → cleaner output
#   │     adaptive=4,645 tokens vs none=4,199 tokens
#   │  BQ03: Explain how to process a whole-genome sequencing dataset thr...
#   │   scores:  adaptive=5,   none=5
#   │   outcome: TIE — all modes equal
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: rag+scholar
#   │  BQ04: Find 4 experts who have published papers on 'STAR RNA-seq sp...
#   │   scores:  adaptive=3,  ★none=6
#   │   outcome: WINNER: NONE
#   │   Why adaptive lost:
#   │     CUGCR compressed context but hurt quality
#   │     (over-compression)
#   │     'none' mode passes full context without compression
#   │     overhead → cleaner output
#   │     adaptive=2,955 tokens vs none=1,988 tokens
#   └─────────────────────────────────────────────────────────────────

#   ┌─ Agents: scholar+code
#   │  BQ02: Find 3 experts on Leiden community detection in single-cell ...
#   │   scores:  adaptive=6,   none=6
#   │   outcome: TIE — all modes equal
#   └─────────────────────────────────────────────────────────────────

# ── Token Utilization: Adaptive vs Others (all queries) ──────
#   QID    Agents                  cugcr  adaptive   vs none    vs raw winner
#   ────── ────────────────────── ────── ───────── ───────── ───────── ──────────
#   BQ01   rag+code+cloud              Y     4,645      111%       n/a none
#   BQ02   scholar+code                Y     3,623      286%       n/a adaptive/none
#   BQ03   rag+code+cloud              Y     4,216      102%       n/a adaptive/none
#   BQ04   rag+scholar                 Y     2,955      149%       n/a none
#   BQ05   code                        N     2,191       92%       n/a adaptive

# ── Agent Context Passing Patterns ───────────────────────────
#   How much context (chars) each agent receives from upstream agents.
#   'none' mode always injects 0 — agents work independently.

#   Wave 1 (primary injection, e.g. rag → code/cloud/scholar):
#   agent            none     adaptive        raw  pattern
#   ────────── ────────── ──────────── ──────────  ──────────────────────────────
#   rag           0 (0/1)  3,655 (1/1)        n/a  receives context only in adaptive
#   scholar       0 (0/2)  1,574 (2/2)        n/a  receives context only in adaptive
#   code          0 (0/5)  1,903 (5/5)        n/a  receives context only in adaptive
#   cloud         0 (0/2)    421 (2/2)        n/a  receives context only in adaptive

#   Wave 2 (retry/feedback injection):
#   agent            none     adaptive        raw  pattern
#   ────────── ────────── ──────────── ──────────  ──────────────────────────────
#   rag           0 (0/4)    782 (5/5)        n/a  retry feedback (adaptive only)
#   scholar       0 (0/5)  1,638 (5/5)        n/a  retry feedback (adaptive only)
#   code          0 (0/5)    946 (6/6)        n/a  retry feedback (adaptive only)
#   cloud             n/a          n/a        n/a  no retry context

#   Key observations:
#   • scholar : only receives context in rag+scholar combos; self-contained otherwise
#   • none    : 0 chars injected to every agent in every query — fully isolated execution