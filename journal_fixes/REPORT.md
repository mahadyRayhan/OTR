# SciWeaver weakness fixes: status on 2026-11-04

Branch `journal-fixes` (not committed). Original vector indexes are backed up in `journal_fixes/backups/`.

## Blocker

Update (same day): the new Gemini key is valid, but the project's prepaid credit is used up ("429 Your prepayment credits are depleted"); a few single test calls succeeded, then every call failed. OpenAI is unavailable, so all experiments now use Gemini only: system `gemini-3.1-flash-lite-preview`, evaluation judges `gemini-3.1-pro-preview` and `gemini-3-flash-preview` (`Config.EVAL_JUDGES`), strong baseline `gemini-3.5-flash`. Gemini 2.5 models are not available to this account. Because judges and system are all Gemini, same-family self-preference cannot be ruled out; re-judging the answers originally scored by GPT-5.4 partly checks it.

Estimated volume for the full run: about 20M tokens, roughly 3M of them on the Pro judge, most of the rest on flash-lite engine runs.

Also found: when a per-node judge call fails, PoT silently scores the node 0.5 with label "unknown" (`_llm_judge` fallback), so an outage degrades the system without any visible error.

Everything that needs an LLM is written, tested offline, and runs with one command once credit is added:

```bash
journal_fixes/run_all.sh
```

## Leakage check and public test sets (2026-11-05, no LLM calls)

**Leakage check** (`scripts/leakage_check.py`, output in `results/leakage_check.json`).

- **What was scanned:** every agent's knowledge base, 9,164 documents and about 4M words. That covers the RAG corpus, the notebooks, the cloud markdown and the scholar cache's 8,700 abstracts.
- **Benchmarks checked:** 4,459 items from LAB-Bench (4 subsets), PubMedQA, MatSciBench, SciCode, BioCoder and BixBench.
- **The checks:**
  - source DOI, PMID or PMCID matches a knowledge-base document;
  - code repository overlap;
  - benchmark canary strings;
  - shared word 13-grams (the contamination test of Brown et al. 2020).
- **Positive control:** 40 of 40 knowledge-base snippets were detected, which shows the detector works.

**Result.**

- **2 items are contaminated.** Both are LitQA2 questions whose source paper is in the RAG corpus: PMC8584034 (prime editing with MLH1dn), and the LungMAP CellRef paper (found by DOI).
- **Everything else is clean:**
  - no other benchmark has any hit;
  - no canary string appears in any knowledge base;
  - no code repository overlaps.
- **Action taken:** the 2 items are excluded from the test sets.

**Implication for evaluation.** The knowledge bases do not overlap with the public benchmarks, so on LitQA2, PubMedQA and MatSciBench the RAG agent will rarely find the answer in its own corpus. Either report results "as deployed", or also run a benchmark-corpus setting where the benchmark's source documents and distractors are indexed. That second setting is standard for RAG benchmarks such as MIRAGE.

**Test sets** (`scripts/build_testsets.py`, output in `data/testsets/`, details in `data/testsets/README.md`).

- **Per agent:** 1,484 items in 10 sets.
  - RAG: LitQA2, PubMedQA, MatSciBench, ProtocolQA.
  - Code: SciCode, BioCoder, SeqQA.
  - Cloud: nf-core resource sizing to an EC2 instance.
  - Scholar: tool paper DOI, and claim to paper DOI.
- **Full system:** 150 items derived from nf-core/modules. Each item has three graded parts: the paper DOI, the command line, and the CPU, memory and instance.
- **Properties:** every item has an exact gold answer, so no LLM judge is needed. 20% of each set is a dev split. No tool appears in more than one of the nf-core-derived sets.

## Bioinformatics, benchmark-aligned knowledge base (2026-11-05, no LLM calls)

Only 8 public questions (2 LitQA2, 6 BioASQ) have a source paper among the 222 deployed papers, so the deployed corpus alone cannot test retrieval. The deployed knowledge bases are therefore kept as the haystack, and only the documents the bioinformatics test questions need were added, in a separate folder (`~/otr_local/bench_kb/bio/`, 3,141 files).

- 125 LitQA2 source papers in full text
- 1,244 BioASQ gold abstracts, taken from one downloaded file
- 10 BioCoder repositories with the target functions removed
- 246 tool READMEs
- an nf-core analogue table covering 1,414 modules

Test sets are in `data/testsets/bio/`, with details in its README. There are 1,093 items in 8 sets, and every item has an exact gold answer.

**Indexes** (`scripts/build_bench_index.py`, written to `~/otr_local/indexes/bio/`). Each index holds the deployed documents plus the added ones, with the same collections, embedding model and chunk formats as the deployed agents.

| Agent | Total chunks | Deployed | Added |
| --- | --- | --- | --- |
| RAG | 12,174 | 7,640 | 4,534 |
| Code | 13,445 | 4,681 | 8,764 |
| Cloud | 4,055 | 267 | 3,788 |

The agents load these indexes when `SCIWEAVER_INDEX_DIR=~/otr_local/indexes/bio` is set. Without it they load the deployed indexes, so deployed behaviour is unchanged.

**Scoring** (`scripts/scoring.py`, self-tested). Every set is scored without an LLM judge:

- MCQ, with an added "insufficient information" option so precision and coverage can be reported;
- yes/no;
- factoid, matched by whole words;
- DOI;
- structured JSON for the cloud and full-system sets, with per-part accuracy.

BioCoder needs the official test harness, which is not set up yet. Factoid items whose gold answer is a full sentence are scored strictly for all systems.

**Alignment check** (`scripts/retrieval_check.py`, output in `results/retrieval_check_bio.json`). This is plain Chroma search with the raw question text, top 10, with no LLM, no query rewriting and no reranking, so it is a lower bound on what the agents retrieve.

| Set | Gold found in top 10 |
| --- | --- |
| LitQA2 | 99% (source paper) |
| BioASQ | 75% (any gold abstract) |
| BioCoder | 83% (repository), 61% (exact file) |
| Full system | 71% (tool README) |
| Cloud | 32% (tool README) |

The knowledge bases and questions are therefore aligned well enough to test retrieval, except in the cloud set. That set's main gold (CPU and memory) is not in any document and must be inferred from analogue modules, so a low README hit rate is expected there.

Caveat: the LitQA2 haystack is small (the 222 deployed papers plus 125 sources), which makes retrieval easy there. This should be reported.

Next steps:

1. ~~A runner for the per-agent and full-system sets.~~ Done: `journal_fixes/scripts/run_pilot.py` (see the pilot below).
2. ~~Set up the BioCoder harness.~~ Done: `journal_fixes/scripts/biocoder_eval.py` (see pilots v3 to v7 below).
3. Run the systems, one part at a time, each part approved first.

### Pilot on the dev split (2026-11-05, $0.80 in total)

`run_pilot.py` asks full SciWeaver (adaptive) and plain Gemini (`gemini-3.1-flash-lite`, same prompt, no retrieval) the same dev-split questions, interleaved across sets. It prints the running accuracy after every item and stops early on signs of breakage (crashes, unparseable answers, or a falling score well below plain Gemini). Results are in `~/otr_local/runs/pilot_bio_v2/`.

The first two attempts (4 and 2 items) were stopped. They exposed these bugs, all now fixed:

| Bug | Effect | Fix |
| --- | --- | --- |
| Model JSON parsed with `json.loads` | When the user's question carries an output instruction ("end with ANSWER: ..."), the model appends it after the JSON. The NeuroRule planner then failed on every query and fell back to the rule splitter, which sent answer options and the format line to agents as tasks. The per-node judge failed the same way, giving false low fitness and needless repairs. | `gemini_client.parse_json` reads the first JSON value; used at all 7 call sites (decomposers, GoT DAG judge and expansion, diagnostician, profiler, RAG provider). |
| Global retry re-synthesis | Did not see the question, so it dropped the requested format. | The prompt now includes the question. |
| Final AGGREGATE prompt | Ignored the format line buried under long agent outputs. | The format requirement is restated at the end. |
| Factoid scorer | "Vaccinia Ankara" was not matched to "Vaccinia virus Ankara". | Also accepts word-overlap F1 >= 0.8. |

Pilot v2, 25 dev items (5 per set; the scholar sets were left out, see below):

| Set | SciWeaver | Plain Gemini |
| --- | --- | --- |
| LitQA2 | 3/5 (1 abstained) | 1/5 |
| BioASQ | 4/5 | 4/5 |
| SeqQA | 4/5 | 5/5 |
| Cloud | 0/5 | 0/5 |
| Full system | 0/5 | 0/5 |
| All | 0.44 (precision 0.46) | 0.40 |

The running score never fell away. Both systems stayed within 0.05 of each other from item 8 on. SciWeaver takes 52 s per item against 3 s.

Problems found (design, not crashes):

1. **Scholar agent cannot return papers.** It only finds experts, so the two scholar DOI sets cannot be passed. The planner also sends "find papers" sub-tasks to it, and its judge scores those 2/10.
2. **The cloud set has a low ceiling.** Offline check (no LLM): predicting the nf-core label from the 5 most similar modules (one module per chunk) gets the right label 37 to 40% of the time. Always answering `process_medium` gets 38%. Exact CPU and memory scoring therefore cannot separate the systems. SciWeaver over-provisions by a large margin (16 CPU / 64 GB for 1 CPU / 6 GB tools), and the deployed HPC guides it retrieves are the likely cause.
3. **Full-system DOIs: SciWeaver 1/5, plain Gemini 3/5.** Retrieved papers overrode the model's correct memory. The tool READMEs, which hold the DOI for 86 of 132 tools, are only in the cloud index, so the literature agents never see them.
4. **SeqQA needs execution.** The code agent writes scripts but does not run them, so answers come from reasoning over the sequence.

### Pilots v3 to v7 (2026-11-06 to 07, about $3 in total)

All tuning used the dev split and general fixes only. No change looks at test items. Runs are in `~/otr_local/runs/pilot_bio_v*`. SciWeaver runs no code: its code agent suggests code. Only the BioCoder evaluation runs code, inside Docker.

Fixes, in order:

| Problem found | Fix |
| --- | --- |
| Scholar could only find experts (problem 1 above) | Paper mode (`scholar_agent/intent.py`, `find_paper`): queries by tool name, Semantic Scholar plus Europe PMC (title first, then full text by citations), exact-phrase search for quoted passages, and an LLM selector restricted to listed DOIs. Paper lookups are routed to it before any other rule. |
| Selector rejected a tool's paper whose title does not name the tool (ShigEiFinder, named at character 1,113 of the abstract) | The selector sees 1,500 abstract characters instead of 300 and is told to check the abstract for the tool's name. |
| Diagnostician wrote a DOI into the retry query, and the judge scored it 10/10 | The revised query may not state an answer; DOIs are stripped from it; the judge is told it cannot check identifiers. |
| Single-leaf plans skipped synthesis and lost the answer format | Every plan ends in a synthesis that follows the requested format and copies identifiers only from verified outputs. |
| Cloud over-provisioning (problem 2) and inconsistent instances | The agent states `SIZING: cpus memory_gb` from comparable modules' defaults; the instance is looked up in the EC2 catalog (`CATALOG-VERIFIED INSTANCE`). Sizing goals are routed to the cloud agent. |
| RAG deduplication kept one chunk per paper | Chunks are deduplicated per passage, with at most 3 per paper. |
| Goals said "the provided sequence" or "options A-D", but agents received only their goal text | When a goal refers to material in the request (provided, given, options, ...), the engine appends the request as reference (`context_format.with_request`). The request's answer-format line is marked as being for the final answer: attaching it to every goal made each cloud node write its own competing JSON. Retrieval ignores the attached request. |
| The planner wrote goals to execute scripts and to re-fetch data the question gives | Planner rules: agents cannot run code; a computation on given data is one goal, routed to the code agent, which adds a hand-worked result after the script. |
| Gold DOIs | Preprint golds accept the published version, and published golds accept their preprint versions (bioRxiv/medRxiv API, Crossref; 126 items gained one). Items whose only gold is a software archive were dropped (5 per set). Log: `<set>.refine_log.json`. |

BioCoder harness (`biocoder_eval.py`, image `docker/sandbox`, shared with `orchestrator/sandbox.py`): a re-implementation of BioCoder's fuzz test. Each item's context file has random inputs filled in five times, and an answer passes if its output matches the golden function's output every time. Runs have no network, a read-only file system except `/tmp`, 1 CPU, 2 GB and 30 s each. Self-test on all 86 items: the golden function passes on 84 and a wrong function fails on all 86. On the other 2 (test split) the golden code crashes on its own fuzzed input; they count as unscorable, not wrong. Dropped 8 items whose prompt names a different function from the one the test calls (86 to 78; `code_biocoder.refine_log.json`).

Results:

| Run (dev) | SciWeaver | Plain Gemini |
| --- | --- | --- |
| v5, 35 items (5 per set) | 0.63 | 0.31 |
| v7, the same 35 items, after all fixes | 0.63 (no abstentions) | 0.31 |
| v7, the 15 paper-lookup items rerun after the selector fix | 0.67 | 0.07 |
| BioCoder, 8 items | 3/8 | 3/8 |
| SeqQA, the same 5 items, three runs | 2/5, 3/5, 3/5 | 4/5 each time |

v7 by set: LitQA2 5/5 vs 1/5, tool paper 5/5 vs 1/5, claim paper 3/5 vs 0/5 (4/5 in the rerun), cloud 3/5 vs 1/5, BioASQ 3/5 vs 4/5, full system 1/5 vs 0/5, SeqQA 2/5 vs 4/5. Full-system parts in the rerun: DOI 4/5, command 4/5, cloud instance 5/5, cloud CPU and memory 2/5.

Remaining limits:

1. **Cloud sizing varies between runs.** The same tool was sized 12/72, 8/16 and 8/32 in three runs, so cloud and full-system scores move by about one item per five between runs. The nf-core labels have a low ceiling (problem 2).
2. **SeqQA.** The sequence now reaches the agents, but tracing a 1,000-base sequence by hand is unreliable. In one item the first agent found the right count and the code agent's trace replaced it with sites that do not exist. Without execution there is no general fix. Report SeqQA separately as a limitation.
3. **BioCoder** failures are shared by both systems. The summary prompts omit details that the tests check (e.g. clipping at 1.0, a custom input class). The code agent neither helps nor hurts here.

## Shared core (2026-11-08, branch `shared-core-Sciweaver`)

Built once and used by both follow-up branches: the journal paper (C1-C6) and the D1 conference paper. Default behaviour is unchanged: the cugcr policy is the old context routing, and the cache is off.

| Part | Files | What it does |
| --- | --- | --- |
| S1 Run trace | `orchestrator/trace.py`, engine and agent hooks | One JSON file per run in `~/otr_local/traces`. For every node it records the task, the exact agent input, the atoms offered and passed on its incoming edge, retrievals, LLM calls (judge calls under `<node>/judge`), tool calls, output, judge score and time. Also the decomposition, the first-pass answer and the retries |
| S2 Replay and cache | `orchestrator/replay.py`, `orchestrator/llm_cache.py` | `SCIWEAVER_LLM_CACHE=1` caches every Gemini call and every live paper search. `replay()` reruns a recorded DAG with changed node outputs or inputs. Checked: an unchanged replay gives the identical answer with 15 of 15 calls from the cache; overriding one node changes only its 9 descendants |
| S3 Context layer | `orchestrator/context_layer.py` | Typed atoms: code, entity, prose, CONSTRAINT (protected), ARTIFACT (passed by pointer), sensitive flag. Policies `none`, `raw`, `cugcr`; a branch adds its own with `@register_policy`. `route_context()` records each edge in the trace |
| S4 Sandbox and lookups | `orchestrator/sandbox.py`, `orchestrator/lookups.py`, `docker/sandbox/` | Model-written Python runs in Docker with no network, a read-only file system except `/tmp`, 1 CPU, 1 GB and a time limit. NCBI E-utilities and remote BLAST run on the host, rate-limited, cached and traced. The BioCoder harness uses the same image |
| S5 Evaluation | `scoring.py`, `run_pilot.py`, `build_geneturing.py` | 95% bootstrap intervals; paired differences with a sign test; `--set KEY=VALUE` overrides for ablations (`CONTEXT_POLICY`, `ATTACH_REQUEST`, and later per-branch switches); a new set `lookup_geneturing` (500 items, GeneGPT's scoring rules) |
| S6 Baselines | `run_pilot.py --systems` | `sciweaver`, `react` (same agents, linear, raw context), `gemini`, `strong` (`gemini-3.5-flash`), `self_consistency` (5 samples, majority vote). Every system's cost is recorded |

Tests: `journal_fixes/scripts/test_shared_core.py [--online]` (offline, plus sandbox when Docker runs and NCBI with `--online`) and `test_replay_live.py <trace>`.

Not done: LAB-Bench2 (SeqQA2 and LitQA3) is gated on Hugging Face (`EdisonScientific/labbench2`, CC BY-SA 4.0). It needs the account owner to accept its terms.

## Part 1 results (2026-11-05): re-judging the stored answers

Run on Vertex AI (project `ai-tutor-rayhan`). The 240 stored answers that GPT-5.4 originally scored were re-scored, in full and untruncated, by `gemini-3.1-pro-preview` and `gemini-3-flash-preview`, using the same rubric. All 240 items were scored by each judge with no errors. Files: `results/rejudge_<judge>.json` (summary), `.jsonl` (per item), `rejudge_usage.json`.

Global Score difference, SciWeaver minus the other system. A positive value means SciWeaver is ahead.

| Query set | vs | GPT-5.4 (original) | Gemini 3.1 Pro | Gemini 3 Flash |
| --- | --- | --- | --- | --- |
| 30 selected queries | ReAct with context | -0.05 | +0.10 (p = 0.60) | +0.07 (p = 0.94) |
| 30 selected queries | ReAct no context | +0.21 | +0.29 (p = 0.86) | +0.31 (p = 0.27) |
| all 50 queries | Gemini (single call) | +0.15 | +0.32 (p = 0.065) | +0.60 (p = 0.005) |
| all 50 queries | gpt-4o (single call) | +0.86 | +1.85 (p < 0.001) | +1.54 (p < 0.001) |

What this shows:
1. **SciWeaver vs ReAct with context is a tie under all three judges.** The sign flips between judges, but every difference is under 0.11 and no p-value is below 0.5. The paper cannot claim a win over this baseline.
2. **The win over gpt-4o holds under all three judges.** It roughly doubles under the Gemini judges. Part of that increase may be family self-preference, since gpt-4o is the only non-Gemini system.
3. **The lead over single-call Gemini grows under Gemini judges** and becomes significant with Flash. Both systems are Gemini, so self-preference affects them equally. This is the most credible positive result so far, but it needs the matched Table 1 (part 4) with the stronger baseline, `gemini-3.5-flash`.
4. **The judges agree only moderately.**
   - Item-level Spearman with GPT-5.4 is 0.67 (Pro) and 0.75 (Flash) on the 30 queries, and 0.53 and 0.45 on all 50.
   - Pro and Flash agree with each other at 0.80 on the 30 queries and 0.60 on all 50.
   - On all 50 queries, correctness agreement between Pro and Flash is only 0.35.
   - Single-judge scores are therefore noisy. Report both judges and their agreement, and do the human agreement study.
5. **Absolute scores are not comparable across judges.** The Gemini judges score 1.3 to 2.3 points higher than GPT-5.4 on the same answers. Compare systems only under the same judge.

Cost (measured with the usage meter):

| Judge | Calls | Tokens | Cost |
| --- | --- | --- | --- |
| Gemini 3 Flash | 240 | 0.28M input, 0.26M output | about $0.90 |
| Gemini 3.1 Pro (137 metered calls) | 137 | 0.12M input, 0.30M output | $3.80 |

Extrapolated to all 240 Pro calls, part 1 cost about **$7.50** in total. Most of the Pro output is thinking tokens.

**OneDrive problem.** OneDrive synced the result files while they were being appended. It created conflict copies, and the main files lost 2 to 3 lines each. The full set was recovered by merging the copies, whose scores were identical where they overlapped. The copies are kept outside the project. Before the next part, results must be written outside OneDrive and copied in afterwards.

## Critical findings from auditing the stored results (no LLM needed)

Script: `journal_fixes/scripts/audit_table1.py`, `journal_fixes/scripts/weight_sensitivity.py`. Outputs in `journal_fixes/results/`.

1. **The 30-query benchmark was selected by outcome.** `best_questions.json` is exactly the 30 of 50 queries where SciWeaver had the largest advantage over the baselines (`results/full_comparison`). On those 30, SciWeaver leads the best baseline by +0.60 Global Score; on the 20 excluded queries it trails by 0.58; on all 50 the lead is +0.13 (not significant, p = 0.17).
2. **Table 1 mixes runs.**
   - GPT-5 and Gemini rows: the Global Score (4.43, 3.88) comes from `results/llm_baseline_comparison` (50 queries), while the dimension scores come from the best-30 subset of `results/full_comparison`, whose "GPT" baseline was `gpt-4o`, not GPT-5. That is why those rows do not satisfy the Global Score formula.
   - "SciWeaver with PoT 6.77": no stored result reproduces it. The stored full-PoT run on the same 30 queries (`results/eval_adaptive.json`) scores 6.26.
   - "SciWeaver without PoT 6.31": top 30 runs by node quality from a pool (`eval_best_adaptive.py`), and only 16 of its 30 queries are benchmark queries.
3. **Matched comparisons (same queries, same judge) do not support the headline claim.**

| Comparison | n | Global Score diff | 95% CI | p | Correctness diff |
| --- | --- | --- | --- | --- | --- |
| SciWeaver (PoT) vs ReAct with context | 30 | -0.05 | [-0.23, +0.14] | 0.58 | 0.00 |
| SciWeaver (PoT) vs ReAct no context | 30 | +0.21 | [-0.04, +0.49] | 0.24 | +0.20 |
| SciWeaver vs GPT-5 (llm_baseline run) | 50 | -0.65 | [-0.86, -0.45] | <0.001 | -1.76 |
| SciWeaver vs Gemini (llm_baseline run) | 50 | -0.11 | [-0.29, +0.07] | 0.25 | -0.60 |
| SciWeaver vs gpt-4o (full_comparison run) | 50 | +0.86 | [+0.64, +1.10] | <0.001 | +0.42 |
| SciWeaver vs Gemini (full_comparison run) | 50 | +0.15 | [-0.05, +0.35] | 0.17 | -0.24 |

   The only robust win is over `gpt-4o`. Weight choice does not rescue the ReAct comparison: ReAct with context wins under 99% of 10,000 random weightings.
4. **The system was optimised against its evaluator.** The in-loop final judge that drives global retries (`_gpt_as_judge`) used `Config.JUDGE_MODEL` (gpt-5.4), the same model that scores the final answers. Fixed (see below); results produced before the fix carry this bias.
5. **Answers were truncated before judging.** `scripts/eval/eval_mode.py` passes only the first 5,000 characters of each answer to the judge. The new scripts pass the full answer.

## What was fixed

| Weakness (doc table) | Fix | Evidence / status |
| --- | --- | --- |
| No cost-matched comparison | `orchestrator/usage_meter.py` counts calls and tokens for every Gemini (both SDKs) and OpenAI call; every run's telemetry now has `llm_usage` | Done; engine smoke test passes. Measurement waits on keys |
| Paper and code differ | Judge prompt now receives the downstream consumers' queries Q(i to j); emits the paper's labels (specificity, coverage, relevance, knowledge_boundary); labels now gate expansion (they were defined but never used); per-type repair hints from Section 4.2.3; tau is `Config.PER_NODE_TAU = 0.70`; unrepaired sub-tasks are surfaced as "Low-confidence parts" and flagged `low_confidence` in telemetry (calibrated abstention was described in the paper but not implemented) | Done; `journal_fixes/scripts/test_pot_alignment.py` passes |
| Judge optimised against itself | New `Config.IN_LOOP_JUDGE_MODEL` (default: the Gemini generation model) for in-run retries; `JUDGE_MODEL` is evaluation only | Done |
| Hand-picked / AI-generated knowledge | RAG: 222 open-access CC-licensed full texts from Europe PMC (161 bioinformatics, 30 materials, 31 health) with real metadata and URLs; index 377 to 7,640 chunks. Notebook agent: 206 tutorial notebooks from scverse, theislab, materialsproject, MONAI (9 before); index 4,681 chunks. Cloud agent: official AWS Price List catalog (1,249 on-demand rows, 168 families, offer 20261104210305) and verbatim tool requirements from official sources (nf-core configs, AlphaFold, BWA-MEM2, Kraken 2, DeepVariant, nnU-Net, LAMMPS, MACE and more); the old unsourced files now carry a provenance line | Done; indexes rebuilt and retrieval spot-checked |
| Heuristic weights | Sensitivity analysis over paper weights, equal weights, single dimensions and 10,000 random weightings | Done (finding 3) |
| Tiny private benchmark | LAB-Bench (LitQA2, ProtocolQA, SeqQA, DbQA) downloaded; judge-free runner with exact-match accuracy, precision, coverage, bootstrap CI and token cost | Ready; needs keys |
| Weak baselines | Self-consistency (k=5, majority vote) on LAB-Bench; Self-Refine (2 rounds) and direct GPT on the 50 queries; raw-context routing ablation | Ready; needs keys |
| Judge not validated | (a) re-judge all 240 stored answers with a different model family and test whether conclusions flip; (b) fault injection with known ground truth for the four failure labels (detection rate, label accuracy, confusion matrix) | Ready; needs keys |
| Profiler fragile, leakage | 800 real researcher questions (Bioinformatics and Matter Modeling Stack Exchange); dual-family LLM labels with kappa; held-out evaluation of the original 32-phrase model vs retrained models; 150-question human annotation sheet | Data ready; labels need keys or annotators |

## Still needs people

- **Human agreement study.** Fill `journal_fixes/data/profiler_human_annotation.csv` (2 annotators); `profiler_eval.py evaluate` uses it as the test set automatically. The same protocol should be applied to 100 to 200 final answers and node outputs (3 domain experts, kappa per criterion).
- **Decide how to report the audit.** The accepted ICDCN numbers are not reproducible from the stored results. The journal version should use the matched, all-50-query, two-judge Table 1 that `run_matched_table1.py` produces.
- **Licence check.** GitHub reports no licence for `theislab/single-cell-best-practices`; confirm before redistributing those notebooks.

## Expected cost of run_all.sh

| Step | LLM calls (approx.) | Wall time (approx.) |
| --- | --- | --- |
| 1 Judge swap | 240 | 5 min |
| 2 Fault injection | 300 | 5 min |
| 3 Profiler labels | 1,600 | 10 min |
| 4 Table 1, cheap systems + judging | 400 + 600 | 20 min |
| 4 Table 1, engine systems (150 runs) | engine | 4 to 7 h (runs one at a time) |
| 5 LAB-Bench (100 items x 4 systems) | 600 + engine | 3 to 4 h |

## Files

- Scripts: `journal_fixes/scripts/` (collectors, audit, sensitivity, runners, tests)
- Results so far: `journal_fixes/results/audit_table1.json`, `journal_fixes/results/weight_sensitivity.json`
- Data: `journal_fixes/data/` (AWS catalog CSV, LAB-Bench subsets, 50-query set, profiler questions and annotation sheet)
- Logs: `journal_fixes/logs/`
