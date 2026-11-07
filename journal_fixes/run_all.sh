#!/usr/bin/env bash
# Runs every LLM-dependent experiment prepared on 2026-11-04, in dependency order.
# Prerequisite: Vertex AI settings in .env (GOOGLE_GENAI_USE_VERTEXAI etc.; see gemini_client.py). All models are Gemini.
# Each step is resumable: rerunning skips finished items.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/agent/bin/python
mkdir -p journal_fixes/logs

echo "== 0. key check"
$PY journal_fixes/scripts/_test_meter.py

echo "== 1. re-judge stored answers with both evaluation judges (2 x 240 calls)"
for J in $($PY -c 'import config;print(" ".join(config.Config.EVAL_JUDGES))'); do
  $PY journal_fixes/scripts/rejudge_family_swap.py --judge-model "$J" | tee "journal_fixes/logs/rejudge_$J.log"
done

echo "== 2. failure-label validation by fault injection (60 nodes x 5 variants = 300 judge calls)"
$PY journal_fixes/scripts/fault_injection.py --n 60 | tee journal_fixes/logs/fault_injection.log

echo "== 3. profiler: dual-LLM labels on 800 real questions, held-out evaluation"
$PY journal_fixes/scripts/profiler_eval.py label
$PY journal_fixes/scripts/profiler_eval.py evaluate | tee journal_fixes/logs/profiler_eval.log

echo "== 4. corrected Table 1 on all 50 queries (cheap systems first)"
$PY journal_fixes/scripts/run_matched_table1.py answer --systems direct_gemini,direct_strong,self_refine
$PY journal_fixes/scripts/run_matched_table1.py answer --systems react_context,sciweaver,sciweaver_raw
$PY journal_fixes/scripts/run_matched_table1.py judge
$PY journal_fixes/scripts/run_matched_table1.py table | tee journal_fixes/logs/table1.log

echo "== 5. LAB-Bench, judge-free (100 questions: 25 per subset; engine runs take ~2 min each)"
$PY journal_fixes/scripts/run_labbench.py --per-subset 25 --systems direct,self_consistency
$PY journal_fixes/scripts/run_labbench.py --per-subset 25 --systems react_context,sciweaver
$PY journal_fixes/scripts/run_labbench.py --summarize | tee journal_fixes/logs/labbench.log
