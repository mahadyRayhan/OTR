# Literature review: verification, LLM-as-a-judge, failure attribution, and calibrated abstention for agentic and multi-agent systems (as of 2026-11-03)

**Scope.** I opened each paper below on its arXiv abs or HTML page, or its PDF. For KnowNo I extracted the PDF text to read its Discussion section. Most papers do not state their venue on the arXiv page.
- "Venue verified" means I confirmed it from the arXiv comments field, PMLR, ICLR proceedings or neurips.cc.
- Anything else is marked **[venue unverified]**.
- Where a fetch was truncated before the limitations section, I say so.

**Reading the limitations.** The limitations and future-work text is a close paraphrase of each paper. "None stated" means the paper has no such section. Sometimes I inferred limitations from the body text, and I mark those as inferred.

---

## A. LLM-as-a-judge: reliability, bias and calibration

### 1. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **Citation:** Zheng et al. NeurIPS 2023 Datasets & Benchmarks [venue unverified; the arXiv page lists none]. https://arxiv.org/abs/2306.05685
- **Core idea:** Uses strong LLMs (GPT-4) as scalable proxies for human preference. Introduces MT-Bench (80 multi-turn questions) and Chatbot Arena (about 30K votes).
- **Results:**
  - GPT-4 agrees with humans 85% of the time (first-turn, no ties), versus 81% between humans.
  - Position consistency when answer order is swapped: GPT-4 65.0%, GPT-3.5 46.2%, Claude-v1 23.8%.
  - Verbosity attack ("repetitive list"): fooled Claude-v1 and GPT-3.5 91.3% of the time, GPT-4 8.7%.
  - Self-enhancement: GPT-4 favours itself by about 10%, Claude-v1 by about 25%.
  - Reference-guided grading cut math grading failures from 70% to 15%.
- **Limitations:** Covers helpfulness only (not safety or honesty). Accuracy, relevance and creativity are merged into one score. The position-bias and self-enhancement studies are limited in scope.
- **Future work:** Open-source judges aligned with humans; broader benchmark categories; better math and reasoning in open models.
- **Relevance to SciWeaver:**
  - Swap-order and reference-guided grading are now baseline mitigations reviewers expect.
  - The weak math/reasoning grading result applies directly to grading bioinformatics code and statistics with an LLM judge.
  - The PoT judge and the final judge should use reference-guided grading wherever ground truth exists.

### 2. LLMs instead of Human Judges? A Large-Scale Empirical Study across 20 NLP Evaluation Tasks (JUDGE-BENCH)
- **Citation:** Bavaresco et al. ACL 2025 main (venue verified via arXiv comments). https://arxiv.org/abs/2406.18403
- **Core idea:** 20 human-annotated datasets used to test whether 11 LLMs reproduce human judgments.
- **Results:** Reliability varies widely across models, tasks and evaluated properties. It also depends on annotator expertise and on whether the text is human- or model-written. The abs page gives no numbers.
- **Limitations:** No explicit section on the abs page. Its main recommendation is that LLM judges must be validated against human judgments before use.
- **Relevance to SciWeaver:** This is the norm-setting citation for "validate your judge per property". SciWeaver uses six final-answer criteria and four PoT criteria, and each one needs its own human-agreement number.

### 3. LLM Evaluators Recognize and Favor Their Own Generations
- **Citation:** Panickssery, Bowman, Feng. NeurIPS 2024 [venue unverified; the arXiv page lists none]. https://arxiv.org/abs/2404.13076
- **Core idea:** Self-preference bias is linked to self-recognition.
- **Results:**
  - GPT-4 and Llama 2 can recognise their own outputs at non-trivial accuracy.
  - After fine-tuning, self-recognition correlates linearly with the strength of self-preference.
  - Controlled experiments suggest the link is causal.
- **Limitations:** None stated on the abs page. The causal claim only "resists straightforward confounders", which implies subtler confounders were not ruled out.
- **Relevance to SciWeaver:** If the PoT judge and the specialist agents come from the same model family, the fitness scores are suspect. Reviewers will ask for judge-family swaps.

### 4. Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge (CALM)
- **Citation:** Ye et al. ICLR 2025 (venue verified via ICLR proceedings). https://arxiv.org/abs/2410.02736
- **Core idea:** An automated perturbation framework that measures 12 judge biases:
  - position, verbosity, compassion-fade, bandwagon, distraction, fallacy oversight;
  - authority, sentiment, diversity, chain-of-thought, self-enhancement, refinement-awareness.
- **Metrics:** Robustness Rate and Consistency Rate.
- **Results:**
  - Claude-3.5 was the most resilient overall, but no model was best on every bias.
  - Position bias grows with the number of candidates (robustness falls below 0.5 at 3–4 options).
  - Self-enhancement bias is significant.
  - Fake quotes and book references sway judges more than fake URLs.
  - Bias is stronger on alignment-style data than on factual data.
- **Limitations:** No dedicated section in the portion I read. The conclusion positions CALM as a tool for future bias-mitigation standards.
- **Relevance to SciWeaver:** The authority and fake-citation results matter for the "Grounding" criterion, which judges citations in bioinformatics answers. CALM-style perturbation tests are a cheap robustness appendix for the PoT judge.

### 5. JudgeBench: A Benchmark for Evaluating LLM-Based Judges
- **Citation:** Tan et al. ICLR 2025 [venue unverified]. https://arxiv.org/abs/2410.12784
- **Core idea:** 350 response pairs with objective correctness labels, built from MMLU-Pro, LiveBench and LiveCodeBench. Covers knowledge (154), reasoning (98), math (56) and coding (42).
- **Results:**
  - GPT-4o with a vanilla prompt scores 50.86%, about random (50%).
  - Most fine-tuned judges score below random; the best (Skywork Llama-3.1-70B) reaches 57.43%.
  - Claude-3.5-Sonnet scores 64.29%; o3-mini (high) scores 80.86%.
- **Limitations:** Pairs are biased toward the generating model (Claude-3.5 drops to 44.8% on its own pairs). Fine-tuned judges produce many ties or invalid outputs and are inconsistent across swapped orderings.
- **Future work:** Reasoning-enhanced judges.
- **Relevance to SciWeaver:** For "Correctness" and "Code Quality", an LLM judge without strong reasoning is close to a coin flip on hard items. Code execution should be the primary signal, with the LLM judge used only as a fallback.

### 6. No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding
- **Citation:** Krumdick et al. Preprint [venue unverified]. https://arxiv.org/abs/2503.05061
- **Core idea:** Judges agree with experts only when they can answer the question themselves, or when they are given a correct reference.
- **Data:** BFF-Bench (160 finance questions) and VERDICTS (1,200 expert-judged responses).
- **Results:**
  - On questions GPT-4o answered correctly, judge–human κ was 0.69 with a human reference, 0.46 with no reference, and 0.21 with a wrong reference.
  - When GPT-4o graded its own answers to questions it could not answer, κ fell to 0.16 (versus 0.86 on questions it answered correctly).
  - 37.5% of MT-Bench's GPT-4 reference answers were wrong.
- **Limitations:** Covers only finance and math. The pairwise setup drops pairs where both or neither answer is correct. Expert annotation is costly. Synthetic references may increase self-preference.
- **Future work:** Judging methods that are both reliable and scalable.
- **Relevance to SciWeaver:** This is the key argument against an ungrounded PoT judge. A failing "knowledge-boundary" node is exactly where the judge also lacks knowledge, so judge reliability is lowest where SciWeaver depends on it most.

### 7. Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement
- **Citation:** Jung, Brahman, Choi. ICLR 2025, oral (venue verified). https://arxiv.org/abs/2407.18370
- **Core idea:** Selective evaluation. A judge's verdict is trusted only above a confidence threshold, calibrated by fixed-sequence testing on a small human-labelled set.
  - Simulated Annotators: a confidence measure built from few-shot simulated annotators and their agreement.
  - Cascaded Selective Evaluation: Mistral-7B, then GPT-3.5, then GPT-4, escalating only when needed.
- **Results:**
  - Guarantees over 80% human agreement on ChatArena with about 80% coverage (79.1%). 88.1% of evaluations are handled by the cheaper models.
  - On TL;DR at a 0.9 target, the guarantee holds in 90.8% of runs, versus 47.5% for GPT-4 alone.
  - Simulated Annotators cuts ECE by about 50% and raises AUROC by about 13%.
- **Limitations** (inferred; no dedicated section):
  - Needs an exchangeable calibration set; coverage drops under distribution shift.
  - Pairwise preference only.
  - Tested on three datasets.
- **Relevance to SciWeaver:** This is the closest template for turning PoT's threshold tau into a statistically calibrated per-node gate. "Calibrated abstention" is currently a heuristic; this paper shows how to make it a guarantee.

### 8. Overconfidence in LLM-as-a-Judge: Diagnosis and Confidence-Driven Solution
- **Citation:** Tian et al. 2025. Preprint [venue unverified]. https://arxiv.org/abs/2508.06225
- **Core idea:** LLM judges are systematically overconfident. Proposes:
  - TH-Score, which measures confidence–accuracy alignment in the high- and low-confidence regions;
  - LLM-as-a-Fuser, a model that fuses several judges' verdicts and critiques.
- **Results** (JudgeBench ECE):
  - GPT-4o 39.25, Llama-3.3-70B 47.37, Mistral-Nemo 74.22.
  - Best single models: Qwen3-235B 11.78 (77.4% accuracy) and DeepSeek-R1 12.07.
  - The fuser reaches 86.29% accuracy at ECE 6.42.
- **Limitations:** No dedicated section. TH-Score is sensitive to its threshold ε. Evaluated on JudgeBench only.
- **Future work:** Root causes of overconfidence; more scalable fixes.
- **Relevance to SciWeaver:** Gives concrete evidence that an uncalibrated judge's fitness score should not be read as a probability. It also offers a multi-judge fusion baseline.

### 9. Rethinking Verbalized Confidence for LLM-as-a-Judge: A Compatibility Shift on Post-2025 Proprietary Models
- **Citation:** Hsiao. arXiv, Sep 2026 [venue unverified]. https://arxiv.org/abs/2609.10996
- **Core idea:** On post-2025 flagship models, verbalized 0–100 confidence now beats logprobs as a soft judge score. The recipe adds an overconfidence advisory and single-call self-debate.
- **Results:**
  - Sonnet 4.5 on AggreFact: AECE falls from 8.9% to 3.8%.
  - 67 of 90 model–task comparisons favour the recipe; none favour the baseline.
  - Pre-2025 models lose balanced accuracy under the recipe.
- **Limitations** (no dedicated section; drawn from scope):
  - Pointwise, single-call judging only (no pairwise or multi-agent judging, no code or math).
  - The logprob comparison covers only GPT-family models.
  - The mechanism is untested.
  - Prompt–model compatibility is a moving target.
- **Relevance to SciWeaver:** PoT uses pointwise rubric scores, so this is directly applicable. Report soft-score calibration (AECE) as well as accuracy, and note that results may not transfer across judge generations.

### 10. Calibrate, Don't Curate: Label-Efficient Estimation from Noisy LLM Judges
- **Citation:** Li et al. arXiv, May 2026 [venue unverified]. https://arxiv.org/abs/2605.09702
- **Core idea:** Calibrating the full judge panel beats picking the top-k most accurate judges. The pipeline is a one-coin Bayesian vote, bias correction, beta calibration, and optional split-conformal sets.
- **Results:**
  - On RewardBench2, the full panel's NLL is about 0.006, versus about 0.013 for top-5 selection.
  - Top-5 selection raises NLL by 19–123% across JudgeBench, RewardBench, RewardBench2 and LLMBar.
  - The result holds after deduplicating judges by model family.
- **Limitations:** Needs labelled calibration data, so it does not cover open-ended evaluation without ground truth. The one-coin model idealises dependence between judges. Fails when judges are below chance or unparseable.
- **Relevance to SciWeaver:** A cheap path from one judge to a calibrated multi-judge fitness score, requiring only a small expert-labelled calibration set.

### 11. Robust Conformal Consensus: Multi-Agent LLM-as-a-Judge Interval Evaluation with Conformal Prediction
- **Citation:** L. Liu. arXiv, Sep 2026 [venue unverified]. https://arxiv.org/abs/2609.06367
- **Core idea:** Builds conformal prediction intervals for each of several judges' scores and aggregates them, giving more stable uncertainty than a single judge.
- **Results:** No numbers on the abs page [results unverified].
- **Limitations:** None stated on the abs page.
- **Relevance to SciWeaver:** Supports reporting interval-valued fitness scores rather than point scores.

---

## B. Agentic judges and judges of agent trajectories

### 12. Agent-as-a-Judge: Evaluate Agents with Agents
- **Citation:** Zhuge et al. ICML 2025 (venue verified via PMLR v267). https://arxiv.org/abs/2410.10934
- **Core idea:** Gives the judge agentic tools so it can evaluate intermediate requirements, not just the final output.
- **Benchmark:** DevAI, 55 AI-development tasks with 365 hierarchical requirements arranged as a dependency graph.
- **Results:**
  - Agent-as-a-Judge agrees with the human majority on 83.88–92.07% of requirements, versus 60.38–84.15% for LLM-as-a-Judge.
  - Individual humans agree 84.88–89.34% of the time; the human majority vote reaches 93.98–95.08%.
  - About 97.7% saving in cost and time (about $30.58 and 118 minutes, versus about $1,297 and 86.5 hours for humans).
- **Limitations:**
  - Proof of concept; the planning and memory modules were unstable.
  - The search module gave little benefit.
  - Gray-box results need trajectories that are rarely available in practice.
  - Humans disagreed with each other 10–30% of the time.
- **Future work:** Use the judge as a process-supervised reward; judge and agent improving each other; automated optimisation of the judge.
- **Relevance to SciWeaver:**
  - Closest analogue to PoT, since both judge requirements arranged in a dependency structure.
  - Sets the protocol of 3 human experts with majority-vote consensus, and reporting judge agreement against that consensus.
  - SciWeaver should replicate this protocol.

### 13. AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories
- **Citation:** Lù et al. 2025. COLM 2025 [venue unverified]. https://arxiv.org/abs/2504.08942
- **Core idea:** 1,302 expert-labelled trajectories from 4 agents across 5 web benchmarks, used to evaluate 12 LLM judges.
- **Results:**
  - No judge exceeds 70% precision (the best is about 69.8%).
  - Rule-based evaluation has 83.8% precision but 55.9% recall.
- **Limitations and future work:**
  - A 30% false-positive rate makes these judges unsafe for reward modelling.
  - Judges trust the agent's own account of its reasoning, misread grounding, and miss instruction details.
  - Future judges must resist being swayed by the agent's reasoning.
- **Relevance to SciWeaver:** If the PoT judge reads an agent's self-explanation, the "judge trusts agent reasoning" failure applies directly. Judge on artifacts (code output, retrieved records), not on narratives.

### 14. PaperBench: Evaluating AI's Ability to Replicate AI Research
- **Citation:** Starace et al. (OpenAI) 2025. ICML 2025 [venue unverified]. https://arxiv.org/abs/2504.01848
- **Core idea:**
  - 20 ICML 2024 papers with author co-designed rubric trees (8,316 leaf requirements), graded by an LLM judge (SimpleJudge).
  - The judge itself is validated on a separate benchmark, JudgeEval.
- **Results:**
  - Best agent: Claude 3.5 Sonnet at a 21.0% replication score.
  - Human ML PhDs: 41.4% on a 3-paper subset, versus 26.6% for o1.
  - The judge (o3-mini) reaches F1 0.83 on JudgeEval, at about $66 per paper.
- **Limitations:** Only 20 papers. Possible contamination. Rubric creation is costly. The judge is less accurate than a human expert and non-deterministic. Evaluation is expensive.
- **Future work:** Automated rubric creation, better judges, studying specification gaming.
- **Relevance to SciWeaver:** Sets the norm for scientific-agent evaluation: rubrics co-authored with domain experts, plus a separate meta-evaluation set for the judge.

---

## C. Self-verification and critique methods

### 15. Large Language Models are Better Reasoners with Self-Verification
- **Citation:** Weng et al. EMNLP 2023 Findings (venue verified via arXiv comments). https://arxiv.org/abs/2212.09561
- **Core idea:** Backward verification. The chain-of-thought conclusion is fed back as a condition of the original problem, and the candidate whose condition is most consistent with the question is selected.
- **Results:** Gains on arithmetic, commonsense and logical reasoning (the abs page gives no numbers).
- **Limitations:** Not stated on the abs page. The motivation section notes that CoT is sensitive to error accumulation.
- **Relevance to SciWeaver:** A per-node verifier that does not need an external judge. It could add a "backward consistency" signal to the fitness score.

### 16. CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing
- **Citation:** Gou et al. ICLR 2024 [venue unverified]. https://arxiv.org/abs/2305.11738
- **Core idea:** A verify-then-correct loop in which the model critiques its output using external tools (search, Python, Perspective API).
- **Results** (ChatGPT):
  - Free-form QA: +7.7 F1.
  - Math: +7.0 points absolute.
  - Toxicity probability: −79.2%.
  - Self-correction without tools gave modest gains or degraded results.
- **Limitations:** Latency grows linearly with the number of iterations. Depends on hand-written demonstrations. Text only, with a limited set of tasks and models.
- **Future work:** More tools (WolframAlpha, dictionaries), more modalities, fewer hand-written demonstrations.
- **Relevance to SciWeaver:**
  - Bioinformatics has strong external verifiers: BLAST/UniProt lookups, code execution, schema checks.
  - PoT should be tool-grounded rather than purely LLM-introspective.
  - Note that AgenTracer (#27) found CRITIC sometimes degrades multi-agent systems.

### 17. Chain-of-Verification Reduces Hallucination in LLMs (CoVe)
- **Citation:** Dhuliawala et al. (Meta). Findings of ACL 2024 [venue unverified]. https://arxiv.org/abs/2309.11495
- **Core idea:** Draft an answer, plan verification questions, answer them independently (the factored variant), then revise.
- **Results** (Llama-65B):
  - Wikidata list precision: 0.17 → 0.36.
  - MultiSpanQA F1: 0.39 → 0.48.
  - FActScore on biographies: 55.9 → 71.4 (Factor+Revise), above ChatGPT (58.7) and PerplexityAI (61.6).
- **Limitations:** Reduces but does not eliminate hallucination. Covers only directly stated factual errors. Capped by the base model's knowledge. Adds compute.
- **Future work:** Retrieval and tool use inside the verification step.
- **Relevance to SciWeaver:** The factored verification idea suits a "specificity" or "coverage" repair: generate probe questions per claim. That is arguably what "Probing-of-Thought" should cite and contrast against.

### 18. Large Language Models Cannot Self-Correct Reasoning Yet
- **Citation:** Huang et al. ICLR 2024 (venue verified via arXiv comments). https://arxiv.org/abs/2310.01798
- **Core idea:** Intrinsic self-correction, with no external feedback, does not reliably help reasoning.
- **Results** (GPT-3.5):
  - CommonSenseQA: 75.8 → 38.1 after one round.
  - GSM8K: 75.9 → 74.7.
  - Gains in earlier papers relied on oracle labels to decide when to stop.
  - Multi-agent debate is worse than self-consistency at an equal number of responses (GSM8K: 83.2 vs 85.3 at 6 responses; 83.0 vs 88.2 at 9).
- **Limitations:** Focuses on reasoning tasks only.
- **Suggestions:** Use external feedback; report inference cost; compare against self-consistency; invest equally in initial prompts and feedback prompts.
- **Relevance to SciWeaver:** The single most important baseline requirement. The Re-Planner's retries must beat self-consistency or resampling at equal token budget. Otherwise reviewers will attribute the gains to extra sampling.

---

## D. Multi-agent debate as verification

### 19. Improving Factuality and Reasoning in Language Models through Multiagent Debate
- **Citation:** Du et al. ICML 2024 [venue unverified; the abs page lists none]. https://arxiv.org/abs/2305.14325
- **Core idea:** Multiple LLM instances debate over several rounds to reach a consensus answer.
- **Results:** The abs page claims improved math and strategic reasoning and factuality, with no numbers there.
- **Limitations:** None stated on the abs page.
- **Relevance to SciWeaver:** The canonical "verification by debate" baseline that a multi-judge PoT variant would be compared against.

### 20. Debate or Vote: Which Yields Better Decisions in Multi-Agent LLMs?
- **Citation:** Choi, Zhu, Li. 2025. NeurIPS 2025 [venue unverified]. https://arxiv.org/abs/2508.17536
- **Core idea:** Separates the two ingredients of multi-agent debate (MAD). Majority voting explains most of the gain. Under homogeneous agents, debate is a martingale: expected belief in the correct answer does not change.
- **Results:**
  - Qwen2.5-7B: single agent 0.7205, majority vote 0.7691, best MAD 0.7377.
  - Llama3.1-8B: majority vote 0.7242 beats MAD 0.6929–0.6990.
  - Interventions biased toward the majority answer lift vanilla MAD from 0.7112 to about 0.76.
- **Limitations and future work:** Only the simultaneous-talk protocol was tested (not sequential or summariser-led). The theory assumes homogeneous agents; extending it to heterogeneous agents is future work. Closing the gap to the oracle is open.
- **Relevance to SciWeaver:** Any "multi-judge" or "debate" upgrade to PoT must be ablated against simple voting.

---

## E. Process reward models and step-level verifiers for agents

### 21. Process Reward Models for LLM Agents: Practical Framework and Directions (AgentPRM / InversePRM)
- **Citation:** Choudhury. 2025 preprint [venue unverified]. https://arxiv.org/abs/2502.10325
- **Core idea:**
  - Monte Carlo rollouts produce per-turn value targets; a PRM is fit to them; the policy is trained against the PRM with online DPO.
  - InversePRM learns the PRM from expert demonstrations instead of outcome rewards.
- **Results** (ALFWorld, Llama3.2-3B):
  - AgentPRM: 88.1%, or 91.0% with Best-of-16.
  - InversePRM: 86.6%, versus 63.4% for SFT.
  - Both beat GPT-4o ReAct (65.7%).
- **Limitations:**
  - Reward hacking: validation PRM reward kept rising while true success fell, with no reliable way to detect it; PRM ensembles did not fix it.
  - Saturation and over-optimisation; slow exploration in early iterations.
  - Only one benchmark.
- **Future work:** Richer environments; model-predictive reasoning with learned world models.
- **Relevance to SciWeaver:** PoT's fitness score acts as a process reward. If it is ever used for training or as an RL signal, expect the same Goodhart effect: agents will learn to satisfy the judge's rubric.

### 22. AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress
- **Citation:** Xi et al. (Fudan). Nov 2025 [venue unverified]. https://arxiv.org/abs/2511.08325
- **Core idea:** Scores each step on "promise" (chance of reaching the goal) and "progress" (advantage over previous steps). Labels come from TD estimation with GAE instead of Monte Carlo rollouts.
- **Results:**
  - More than 8× more compute-efficient than baseline reward models.
  - Beam search with Qwen2.5-3B on WebShop: 76.0 versus 54.5 for a process value model.
  - Keeps improving as Best-of-N grows.
  - Evaluated on WebShop, BabyAI and TextCraft.
- **Limitations and future work:** No dedicated section in the visible text (truncated). Future work: use it for RL training; extend to coding and logical reasoning.
- **Relevance to SciWeaver:**
  - "Progress" maps naturally to DAG nodes: did this node move the query closer to an answer?
  - A learned PRM could replace or calibrate the hand-weighted fitness score (completeness, depth, accuracy, clarity).

---

## F. Failure taxonomies and trajectory error benchmarks

### 23. Which Agent Causes Task Failures and When? (Who&When)
- **Citation:** S. Zhang et al. ICML 2025 Spotlight (venue verified via PMLR v267). https://arxiv.org/abs/2505.00212
- **Core idea:** Defines automated failure attribution: identify the responsible agent and the decisive error step. Benchmark: 184 annotated failure tasks from 127 multi-agent systems.
  - Algorithm-generated systems: CaptainAgent on GAIA and AssistantBench.
  - Hand-crafted systems: Magentic-One.
  - Three experts spent about 23–31 hours each annotating.
  - Methods tested: all-at-once, step-by-step, binary search.
- **Results:**
  - Best agent-level accuracy 53.5%; best step-level accuracy 14.2%. Some methods score below random.
  - o1 and R1 were not practically usable.
  - Accuracy drops as logs get longer.
- **Limitations:** No dedicated section. The conclusion says the task is very hard and calls for more research. The impact statement flags risks once systems act on external environments.
- **Relevance to SciWeaver:**
  - PoT's per-node judge plus failure type is an online form of agent-level attribution.
  - The step-level accuracy of about 14% shows that LLM attribution is unreliable, so SciWeaver's failure labels need validation against human labels.

### 24. Why Do Multi-Agent LLM Systems Fail? (MAST)
- **Citation:** Cemri et al. (Berkeley). NeurIPS 2025 Datasets & Benchmarks (venue verified via neurips.cc). https://arxiv.org/abs/2503.13657
- **Core idea:** A 14-mode taxonomy in 3 categories (specification/system design, inter-agent misalignment, task verification), built with Grounded Theory. MAST-Data has 1,600+ (1,642) traces from 7 multi-agent frameworks.
- **Results:**
  - Human inter-annotator κ = 0.88.
  - The o1 LLM annotator reached 94% accuracy and κ = 0.77; κ = 0.79 on out-of-domain systems.
  - Interventions on ChatDev: clearer roles +9.4%; an added verification step +15.6%. Gains were partial.
- **Limitations:**
  - Not exhaustive.
  - Some failures stem from base-LLM limits.
  - Closed systems (Manus) excluded.
  - Correlated failure modes (r up to 0.63) can confuse the automatic annotator.
- **Future work:** Structural fixes (verification, communication protocols, uncertainty quantification, memory/state). Better base models alone will not solve the problem.
- **Relevance to SciWeaver:**
  - The κ reporting protocol (human–human and judge–human) is what SciWeaver's four failure labels lack.
  - Mapping SciWeaver's labels onto MAST would add external validity.
  - MAST's "task verification" category is exactly what PoT targets.

### 25. TRAIL: Trace Reasoning and Agentic Issue Localization
- **Citation:** Deshpande et al. (Patronus AI). 2025 [venue unverified]. https://arxiv.org/abs/2505.08638
- **Core idea:** Error taxonomy covering reasoning, system execution, and planning/coordination. 148 OpenTelemetry-style traces from GAIA and SWE-Bench Lite, with 841 annotated errors.
- **Results:**
  - Best model Gemini-2.5-Pro: about 11% joint accuracy (18% on GAIA, 5% on SWE-Bench).
  - 3 of 8 models could not fit the full trace in context.
- **Limitations:** Text-only. Many error categories have very few examples, which hurts judge reliability on rare, high-impact failures.
- **Future work:** Synthetic perturbation of traces to create rare failures; scalable evaluation frameworks.
- **Relevance to SciWeaver:** Span-level localization is a closer analogue to per-node judging than whole-trace attribution. It is also a ready external benchmark to test the PoT judge as a span verifier.

### 26. AgentMonitor: A Plug-and-Play Framework for Predictive and Secure Multi-Agent Systems
- **Citation:** Chan et al. (HKUST). 2024 [venue unverified]. https://arxiv.org/abs/2408.14972
- **Core idea:** Wraps each agent, logs its inputs and outputs, and turns them into LLM-judged scores plus graph features. An XGBoost model predicts system performance. A post-editor rewrites harmful outputs.
- **Results:**
  - Spearman 0.89 in-domain; about 0.58 cross-architecture or cross-task.
  - Post-editing: 6.2% less harmful and 1.8% more helpful content.
- **Limitations:** Post-editing is incomplete, especially with weak editors. Latency from editing every agent. Only 5 architectures and 3 benchmarks.
- **Future work:** Apply the monitor selectively, only when needed.
- **Relevance to SciWeaver:** The closest prior work for per-agent online monitoring combined with graph features. Reviewers will ask how PoT differs. Answer: PoT does targeted repair and DAG rewiring, whereas AgentMonitor only predicts performance or post-edits.

---

## G. Automated failure attribution: follow-ups to Who&When (2025–2026)

### 27. AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?
- **Citation:** G. Zhang et al. Sep 2025 [venue unverified]. https://arxiv.org/abs/2509.03312
- **Core idea:**
  - Training data built by counterfactual replay plus programmatic fault injection: TracerTraj-2.5K.
  - AgenTracer-8B: Qwen3-8B trained with GRPO using a reward that scores format, the correct agent, and closeness to the correct step.
- **Results** on Who&When with ground truth (agent-level / step-level):
  - Hand-crafted: 69.10 / 20.68, versus Claude-Sonnet-4 56.90 / 17.24.
  - Automated: 69.62 / 42.86.
  - Feedback to multi-agent systems gives +4.8 to 14.2% (MaAS, OWL, MetaGPT).
  - Self-Refine and CRITIC were inconsistent and sometimes hurt.
- **Limitations** (no dedicated section; from the text):
  - Absolute accuracy is modest.
  - Ground-truth answers can mislead the tracer.
  - Annotation depends on a DeepSeek-R1 analyser; injected faults are synthetic.
- **Relevance to SciWeaver:** Shows a small trained attributor can beat frontier LLM judges. This is a strong option for replacing the generic PoT judge with a trained, domain-specific node verifier.

### 28. Abduct, Act, Predict (A2P)
- **Citation:** Y. Weng et al. (Westlake). Sep 2025 [venue unverified]. https://arxiv.org/abs/2509.10401
- **Core idea:** One-pass structured causal prompting: abduce the hidden cause, propose a minimal corrective intervention, then simulate the counterfactual outcome.
- **Results:**
  - Who&When step-level accuracy: 47.46% (algorithm-generated) versus 16.67% all-at-once; 29.31% (hand-crafted) versus 12.07%.
  - Removing explicit step numbering drops accuracy from 47.46% to 17.78%.
- **Limitations:** No dedicated section. About 25% more tokens. Tested with only one model (gpt-oss-120b).
- **Future work:** Other diagnostic domains, efficient search strategies, formal reasoning tasks.
- **Relevance to SciWeaver:** The "Act" step (minimal intervention) mirrors the Re-Planner's "enrich context" action. A2P-style counterfactual prediction could decide between enriching and rewiring before spending a retry.

### 29. FAMAS: Who is Introducing the Failure? Spectrum-based failure attribution
- **Citation:** Ge et al. (Nanjing). Sep 2025 [venue unverified]. https://arxiv.org/abs/2509.13782
- **Core idea:** Adapts spectrum-based fault localisation from software engineering. About 20 replays of the failed task, abstraction into agent–action–state triples, then a Kulczynski2-based suspiciousness score.
- **Results:**
  - Agent-level 57.61% and action-level 29.35%, versus the best LLM baseline at 54.59% / 19.68%.
  - Action-level is 49% better relative to the best baseline.
- **Limitations:**
  - About 105 minutes per attribution (248 minutes on hand-crafted systems).
  - Non-monotone in log length.
  - Who&When only.
  - Strict top-1 scoring; sensitive to parameters.
- **Future work:** Hierarchical context modelling and trajectory pruning.
- **Relevance to SciWeaver:** Repeated runs of a DAG give a natural "spectrum". With only 4 nodes, spectrum analysis is cheap and gives a statistical, non-LLM complement to the judge.

### 30. Where LLM Agents Fail and How They Can Learn From Failures (AgentErrorTaxonomy / AgentErrorBench / AgentDebug)
- **Citation:** K. Zhu et al. (UIUC/Stanford). Sep 2025 [venue unverified]. https://arxiv.org/abs/2509.25370
- **Core idea:**
  - A modular taxonomy (memory, reflection, planning, action, system).
  - 200 annotated trajectories from ALFWorld, WebShop and GAIA.
  - AgentDebug: find the earliest critical error via counterfactual testing, then re-roll with targeted feedback.
- **Results:**
  - All-correct detection accuracy 24.3% versus 0.3%; step accuracy 45.0% versus 28.0%.
  - Up to 26% relative task-success gain (e.g., ALFWorld with GPT-4o-mini: 21 → 55).
  - Inter-annotator κ = 0.55. The authors call this "substantial"; by Landis–Koch it is only moderate.
- **Limitations:** Small and narrow benchmark. No multimodal, long-horizon or safety-critical domains. Prompt-only (no trained debugger).
- **Future work:** Extend to those settings; train a dedicated debugging model.
- **Relevance to SciWeaver:**
  - "Error propagation is the primary bottleneck" supports a DAG-aware PoT.
  - "Fix the root cause, not surface errors" is a design principle the Re-Planner should test explicitly.

### 31. Who Broke the System? Failure Localization in LLM-Based Multi-Agent Systems (AgentLocate)
- **Citation:** Xia et al. COLM 2026 (venue from arXiv comments: "To appear"). https://arxiv.org/abs/2607.07989
- **Core idea:** An LLM judge, multiple independent evaluators, confidence-aware aggregation, and lightweight fine-tuning of the judge on the aggregated feedback.
- **Results:**
  - Who&When algorithm-generated, Qwen-7B judge: agent 69.05 / step 38.10, versus AgenTracer 45.24 / 11.90 and ECHO 30.95 / 14.29.
  - Hand-crafted: 62.50 / 16.67.
  - Aegis-Bench: agent 54.83, versus 46.17.
- **Limitations:** No dedicated section. Modest gains on fine-grained error modes. Struggles on long hand-crafted traces.
- **Future work:** Model error propagation explicitly, separating the true upstream source from the point where the failure surfaces.
- **Relevance to SciWeaver:**
  - Confidence-aware multi-evaluator aggregation is a peer-reviewed template for upgrading the single PoT judge.
  - The stated future work (source vs surfacing point) is exactly the DAG-attribution gap.

### 32. Adaptive Influence Graphs (AIG) for Failure Attribution in Multi-Agent Systems
- **Citation:** Bakish et al. arXiv, Aug 2026 [venue unverified]. https://arxiv.org/abs/2608.24361
- **Core idea:** Attribution depends on how the trace is represented. Convert the trace into a structured influence graph, then traverse it with an agent.
- **Results** on Who&When (step / agent accuracy, no ground truth):
  - Opus-5: 55.20 / 71.20 (algorithm-generated); 29.31 / 63.79 (hand-crafted).
  - Baselines: A2P 47.50 / 65.40; CHIEF 45.60 / 68.80.
  - AgenTracer-8B scores 37.30 / 63.73 in this table, lower than its self-reported numbers because the settings differ.
- **Limitations:**
  - Who&When only.
  - The benchmark lacks agent inputs.
  - Intermediate graphs are not scored against references.
  - Extra token cost.
- **Future work:** Benchmarks that release agent inputs and tool logs.
- **Relevance to SciWeaver:** SciWeaver already has an explicit DAG and logs every node's inputs. That is the kind of structured trace AIG had to reconstruct, which is a natural advantage to exploit and claim.

### 33. Who&When Pro: Can LLMs Really Attribute Failures in AI Agents?
- **Citation:** J. Liu et al. (Penn State / AG2). arXiv, Jul 2026 [venue unverified]. https://arxiv.org/abs/2607.09996
- **Core idea:** Labels from controlled error injection into replayed successful runs, giving golden labels.
  - 12,326 traces from 26 benchmarks and 15 frameworks; text, image and video; 18 error modes.
  - Human check on 100 traces: 94% of step labels accepted; Fleiss κ 0.73.
- **Results:**
  - Best text step accuracy about 74% (Qwen3.5-122B); best error-mode F1 only about 22%.
  - Accuracy falls from about 94% under 3K tokens to about 50% above 12K tokens.
  - Planning, verification and coordination errors are mislabelled as reasoning errors.
  - All-at-once beats step-by-step.
  - Giving the judge the ground-truth answer does not help, and can hurt, reasoning-error attribution.
- **Limitations:** A "taxonomy scope" limitation exists in the appendix but was truncated in my read [details unverified]. The conclusion says failure-mode diagnosis is still weak.
- **Relevance to SciWeaver:** Error-*type* classification is the weakest link (F1 about 22%). SciWeaver's four-way labels (specificity / coverage / relevance / knowledge-boundary) are exactly this kind of label and need direct validation.

### 34. Repair or Resample? Rethinking Failure Debugging in LLM Multi-Agent Systems (SymTrace / SymFail)
- **Citation:** Luan et al. (ECNU). arXiv, Aug 2026 [venue unverified]. https://arxiv.org/abs/2608.25920
- **Core idea:** Full reruns confound "repair" with "resampling luck".
  - SymTrace replays the exact prefix and regenerates only after the intervention point.
  - SymFail: 536 human-annotated failures on AG2, CrewAI and Magentic-One.
  - Suspicious-Node Intervention applies one symptom-conditioned repair at the flagged node.
- **Results:**
  - Replay reproduces the failure 80.78% of the time per execution, versus 67.97% for a rerun.
  - Node intervention repairs 20.15% of failures with a single intervention, versus 6.90% for the best rerun baseline (×2.92).
  - Self-reflection and critic regeneration were no better than reruns; reruns can also break previously successful cases.
- **Limitations:**
  - Benchmark tasks only.
  - LLM-judge evaluation (they recommend reporting human–LLM agreement).
  - One model at one temperature.
  - The replay guarantee covers only the prefix.
- **Relevance to SciWeaver:**
  - The single most relevant 2026 paper for the Re-Planner. Repair should be evaluated with prefix replay, against equal-budget resampling.
  - "Symptom-conditioned repair" is close to SciWeaver's failure-type-conditioned enrichment.

### 35. Self-Healing Agentic Orchestrators for Reliable Tool-Augmented LLM Systems
- **Citation:** Suresh Babu & Agrawal. arXiv, May 2026 [venue unverified]. https://arxiv.org/abs/2606.01416
- **Core idea:** A monitor → detect → diagnose → recover → verify loop. Failure classes are mapped to recovery actions (retry, argument repair, retrieval refresh, tool substitution, replanning) under explicit budgets.
- **Results:**
  - 98.8% success, versus 93.8% for full replanning and 70.1% for a static workflow.
  - With one recovery attempt: 94.0% versus 88.2%.
  - Verifier-guided runs had 0% silent failures, versus 13–18% without a verifier.
- **Limitations:** 100 synthetic author-built tasks; simulated tools; proxy cost metrics; small live validation; not production-representative.
- **Future work:** Real APIs, natural failure traces, learned hybrid recovery policies.
- **Relevance to SciWeaver:** Same architecture pattern (failure class → repair action). Its weak evaluation is a cautionary example of what A* reviewers will reject.

---

## H. Credit and blame assignment in multi-agent LLM pipelines

### 36. Who Gets the Reward & Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents
- **Citation:** C.-H. Yang (Argonne). NeurIPS 2025 LAW Workshop (venue from arXiv page). https://arxiv.org/abs/2511.10687
- **Core idea:** Theory only.
  - Successful runs: Shapley credit across agents, refined into signed, credit-conserving message-level rewards.
  - Failed runs: first-error localisation plus preference pairs that credit later repair attempts.
- **Results:** None (conceptual paper).
- **Limitations:**
  - Exact Shapley cost is exponential in the number of agents.
  - Step judges may be unreliable.
  - Heterogeneous teams need careful baseline policies.
  - Attribution drifts under distribution shift.
- **Future work:** Empirical validation; heterogeneous specialist teams.
- **Relevance to SciWeaver:** With 4 agents, exact Shapley needs only 2⁴ = 16 coalitions. SciWeaver can afford exact attribution where others cannot.

### 37. The Trace Is the State: Exact Credit Assignment for LLM Agent Teams (C3)
- **Citation:** Y. Chen et al. arXiv 2026 [venue unverified]. https://arxiv.org/abs/2603.06859
- **Core idea:** If the trace records all state, credit can be estimated by counterfactual continuation: resample one message, continue to the end, compare outcomes. This is unbiased under the trace-is-state condition.
- **Results:**
  - Rank correlation 0.69 against a reference estimate, versus 0.29 for a learned critic.
  - Split-half reliability falls from 0.64 on a 2-agent chain to 0.01 on a 10-agent chain.
  - Training (Qwen3-4B): MATH500 82.8 versus 74.5 for MAGRPO (Holm-corrected p < 0.001).
- **Limitations:** Hidden state breaks the guarantee. Costly. A judge-scored reward adds variance. Tested only on math with Qwen3.
- **Future work:** No dedicated section. The conclusion recommends logging everything each role reads.
- **Relevance to SciWeaver:**
  - A principled ground truth for "which node caused the failure". Validate the PoT judge's blame against C3-style counterfactual credit on a subset.
  - The reliability collapse with depth is a warning for deeper DAGs.

### 38. Semantic Cooperative Games for Contribution Attribution in LLM-Based Multi-Agent Systems (SLIC / Semantic Shapley)
- **Citation:** P. Jiang et al. (NYU). arXiv, Jul–Oct 2026 [venue unverified]. https://arxiv.org/abs/2607.18255
- **Core idea:** Builds a semantic support hypergraph from a single trajectory and computes a Semantic Shapley Value without reruns. It reduces to classic Shapley under standard conditions.
- **Results:**
  - HealthBench (3 agents): Kendall τb 0.814 versus 0.575 for leave-one-out and 0.479 for a holistic LLM judge.
  - 85.7% fewer API calls than exhaustive Shapley (93.3% in another section; inconsistent).
  - 93.1% of recovered support links confirmed by counterfactual replay.
- **Limitations:** Implicit or negated support is hard to recover. Inherits judge and rubric instability. The graph is not unique. Injected degradation is used as a proxy for ground truth.
- **Future work:** Workflow auditing, agent pruning, contribution-aware redesign.
- **Relevance to SciWeaver:** The medical-domain multi-agent setting is the closest domain to bioinformatics. Contribution attribution could justify the Re-Planner's rewiring decisions.

**Not opened [unverified]; titles seen only in search results:**
- SHARP: arXiv 2602.08335
- Counterfactual Credit Policy Optimization: arXiv 2603.21563
- COSAC: arXiv 2604.17693
- CausalFlow: arXiv 2605.25338
- ASCon: arXiv 2608.10646
- "Localizing Emergent Failures… Minimal Repair Families": arXiv 2608.29228
- ECHO, CHIEF, RAFFLES and AEGIS appear only as baselines in #31 and #32.

---

## I. Uncertainty quantification, conformal guarantees and abstention

### 39. Robots That Ask For Help: Uncertainty Alignment for LLM Planners (KnowNo)
- **Citation:** Ren et al. CoRL 2023, oral (venue verified). https://arxiv.org/abs/2307.01928
- **Core idea:** Conformal prediction over multiple-choice plan options. If the prediction set is not a singleton, the robot asks a human. Gives a calibrated task-success guarantee, extended to multi-step tasks.
- **Results** (from the PDF):
  - Reaches the user-specified success level while reducing human help by 10–24% versus baselines.
  - Multi-step hardware at ε = 0.25: help reduced by 14% step-wise and 8% trial-wise.
  - A naive set at ε = 0.25 needs 87% step-wise help.
- **Limitations:** Assumes the environment is fully grounded in text and that actions execute successfully. Assumes humans give faithful help.
- **Future work:** Include perception and policy uncertainty in calibration; model human error; active preference queries; optimise conformal prediction for help-rate metrics.
- **Relevance to SciWeaver:** The canonical "calibrated abstention with a guarantee" for LLM planners. SciWeaver's "surface as low-confidence" is the "ask for help" action, but without the calibration.

### 40. Mitigating LLM Hallucinations via Conformal Abstention
- **Citation:** Abbasi-Yadkori et al. (DeepMind). 2024 [venue unverified]. https://arxiv.org/abs/2405.01563
- **Core idea:** Self-consistency similarity among sampled responses serves as the score. Conformal risk control sets the abstention threshold to bound the hallucination rate.
- **Results** (Gemini Pro):
  - Keeps the hallucination rate within the target.
  - Much less conservative than log-probability scores on long answers (Temporal Sequences); similar on TriviaQA.
- **Limitations:**
  - Guarantee holds in expectation; the high-probability version is looser.
  - The match function needs manual labels.
  - O(k²) extra calls.
  - One model, two datasets.
- **Relevance to SciWeaver:** A direct recipe for calibrating the abstention threshold on a small labelled set. Its self-consistency score could augment or replace the judge's fitness score.

### 41. Language Models with Conformal Factuality Guarantees
- **Citation:** Mohri & Hashimoto. ICML 2024 [venue unverified]. https://arxiv.org/abs/2402.10978
- **Core idea:** Decompose an output into sub-claims and back off the least-confident ones until a conformal correctness guarantee holds.
- **Results:**
  - GPT-4 factuality: FActScore 30% → 80%; NQ 78% → 93%; MATH 75% → 95%.
  - Most claims are retained (about half on FActScore at α = 0.2).
  - 50 inputs per dataset.
- **Limitations:** Marginal rather than conditional guarantees. Small calibration sets cause deviation from the target. Breaks under distribution shift.
- **Future work:** Conformal methods robust to shift; better UQ scores.
- **Relevance to SciWeaver:** "Partial abstention", which drops sub-claims rather than whole answers, fits SciWeaver's per-sub-task low-confidence surfacing better than all-or-nothing abstention.

### 42. UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making
- **Citation:** Duan et al. Jun 2025 [venue unverified]. https://arxiv.org/abs/2506.17419
- **Core idea:** Splits uncertainty into intrinsic (within the current step) and extrinsic (inherited from earlier steps, as mutual information), estimated via pointwise mutual information over sampled trajectories.
- **Results:**
  - AUROC on AgentBench-OS: 0.771 versus 0.748. On StrategyQA: 0.628 versus 0.593.
  - Gains of 2.3–11%.
  - Removing the extrinsic term hurts most.
- **Limitations:** Monte Carlo sampling bias and latency. Reproducibility with closed models.
- **Relevance to SciWeaver:** Uncertainty propagated from upstream DAG nodes should enter each node's fitness score. Currently each PoT node is judged in isolation.

### 43. Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition (MATU)
- **Citation:** T. Chen et al. (ASU/UCR). arXiv, Apr 2026 [venue unverified]. https://arxiv.org/abs/2604.08708
- **Core idea:** Embed each agent's step across N = 10 runs and fit a PARAFAC2 tensor decomposition. The reconstruction loss is the uncertainty score; the method is topology-agnostic.
- **Results:**
  - Camel/MATH AUROC: 0.7089 versus 0.6351 for P(true) (Qwen-7B); 0.7354 versus 0.5421 (Llama-8B).
  - Beats SAUP in most settings.
  - Also useful for model selection.
- **Limitations:** Cost is linear in the number of runs. Depends on the embedding model.
- **Future work:** Semantic interpretation of latent factors.
- **Relevance to SciWeaver:** A black-box, judge-free uncertainty signal per agent that could serve as a second opinion beside the LLM judge.

### 44. Uncertainty Quantification for LLM Agents: A Taxonomy, an Evaluation Protocol, and an Empirical Study
- **Citation:** M. Lin et al. (TCD). arXiv, Sep 2026 [venue unverified]. https://arxiv.org/abs/2609.07395
- **Core idea:**
  - Taxonomy: source × method × pipeline stage, from 120 core papers.
  - Proposes Trajectory-Checkpoint ECE (TC-ECE).
  - Shows that perfectly calibrated step confidences can still give a miscalibrated trajectory when steps are dependent.
  - Conformal exchangeability fails for steps that depend on history.
- **Results** (from the abstract; specific numbers not reached in my read):
  - Agents' self-reported confidence does not consistently beat a simple step-index baseline.
  - Pooling all checkpoints hides overconfidence at later stages.
- **Limitations:** Not a PRISMA-style systematic review. Formal results assume an absorbing-failure convention that does not hold when agents retry or re-plan. The ten open problems in Section 11 were not visible in my read.
- **Relevance to SciWeaver:** The norm-setting 2026 evaluation protocol. Its retry/re-plan caveat applies exactly to the PoT Re-Planner, so SciWeaver can position itself as handling that case.

### 45. Know Your Limits: A Survey of Abstention in Large Language Models
- **Citation:** Wen et al. TACL 2025 [venue unverified]. https://arxiv.org/abs/2407.18418
- **Core idea:** Frames abstention from three perspectives (query answerability, model confidence, human values) and catalogues metrics:
  - abstention precision, recall and F1; coverage; reliable accuracy;
  - effective reliability; coverage-at-accuracy; AUC.
- **Open problems:**
  - Over-abstention; fragility to prompt wording and jailbreaks.
  - Fairness across groups and languages.
  - Benchmarks cover only one perspective.
  - Multi-turn, context-aware abstention is under-explored.
  - Follow-up actions after abstaining.
- **Relevance to SciWeaver:** Supplies the metric vocabulary for evaluating calibrated abstention. "Follow-up actions after abstaining" is literally what the Re-Planner does.

### 46. AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions
- **Citation:** Kirichenko et al. (FAIR). 2025. NeurIPS 2025 D&B [venue unverified]. https://arxiv.org/abs/2506.09038
- **Core idea:** 20 datasets (35k+ unanswerable questions) across 6 scenarios, testing 20 models.
- **Results:**
  - Abstention is unsolved; scale barely helps.
  - Reasoning fine-tuning degrades abstention by about 24% on average.
  - More reasoning budget lowers abstention; the RL-with-verifiable-rewards stage hurts it.
- **Limitations:** English only. Possible contamination. Imperfect LLM judge (noise could inflate abstention).
- **Future work:** Abstention-targeted post-training; more underspecified-context data; dynamic test sets.
- **Relevance to SciWeaver:** If SciWeaver's agents use reasoning models, they will under-abstain. External, system-level abstention such as PoT's is therefore well motivated, but needs measuring on unanswerable bioinformatics queries.

---

## J. Domain benchmark

### 47. BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology
- **Citation:** Mitchener et al. (FutureHouse). 2025 [venue unverified]. https://arxiv.org/abs/2503.00096
- **Core idea:** 50+ real analysis scenarios with about 300 open-answer questions.
- **Results:** Frontier models reach about 17% on open-answer questions and are no better than random in the multiple-choice setting.
- **Limitations:** Not on the abs page.
- **Relevance to SciWeaver:** The obvious external, objectively scored bioinformatics benchmark. It avoids judge circularity and is far larger than 30 queries.

---

## 1. Gaps: open problems as of late 2026

1. **Calibrated per-node verification with guarantees that compose across a DAG.**
   - KnowNo (#39) gives multi-step conformal guarantees for *chains* of multiple-choice decisions.
   - Trust or Escalate (#7) and conformal abstention (#40) calibrate *single* judgments.
   - #44 shows calibrated step confidences do not compose when steps are dependent, that exchangeability fails, and that its formal results assume no retry or re-plan.
   - No method gives a guarantee for a DAG of LLM nodes with a repair loop. SciWeaver's setting is exactly this.
   - Concrete target: conformal risk control on tau per node, plus a DAG-aware composition, such as a union bound or Bonferroni over the paths that reach the final answer.

2. **Attribution on explicit DAGs versus linear logs.**
   - Who&When, AgenTracer, A2P, FAMAS and Who&When Pro (#23, #27–29, #33) all treat traces as sequences.
   - AIG (#32) shows representation drives accuracy but must *reconstruct* the graph.
   - AgentLocate's (#31) stated future work is separating the upstream source from where the failure surfaces.
   - Attribution with a known dependency structure and logged node inputs is unexplored. With 4 nodes, exact counterfactual credit (C3, #37) or exact Shapley (#36, #38) is affordable, so it can serve as ground truth for validating the judge's blame.

3. **Online (pre-failure) verification versus post-hoc attribution.**
   - Almost every attribution paper diagnoses *completed failed* traces.
   - PoT gates *during* execution, where false positives cost retries and false negatives propagate errors.
   - No benchmark measures online gate quality: precision/recall of "this node will cause failure" against downstream outcomes.
   - AgentRewardBench (#13) shows trajectory judges stay below 70% precision.

4. **Error-type diagnosis is the weakest link.**
   - Who&When Pro (#33): error-mode F1 about 22%, with planning and verification errors mislabelled as reasoning.
   - MAST (#24): correlated failure modes confuse the LLM annotator.
   - TRAIL (#25): rare categories are under-sampled.
   - SciWeaver's repair policy is conditioned on four unvalidated labels. Open question: does label-conditioned repair beat label-agnostic repair at all?

5. **Repair versus resample, and the equal-budget problem.**
   - Huang et al. (#18): self-correction and debate lose to self-consistency at equal cost.
   - Repair or Resample (#34): full reruns conflate repair with sampling luck; critic and reflection regeneration are no better than reruns; prefix replay is needed.
   - AgenTracer (#27): CRITIC can hurt.
   - Learned repair policies are open. Neither the choice between enriching context and rewiring the DAG, nor when to stop and abstain, has been learned from data anywhere (#35 lists learned recovery policies as future work).

6. **Judge reliability exactly where it is needed most.**
   - No Free Labels (#6): judges agree with experts only on questions they can answer themselves.
   - Knowledge-boundary failures in specialised domains are therefore the cases the judge is least able to detect.
   - JudgeBench (#5): near-random judging on hard items.
   - Overconfidence (#8): ECE of 39 for GPT-4o.
   - Self-preference (#3) and generator bias (#5) add to this. Domain-grounded judging (references, tools, executable checks) for scientific agents is unsolved.

7. **Goodhart effects when the judge drives control or training.**
   - AgentPRM (#21): undetectable reward hacking.
   - AgentRewardBench (#13): judges trust agent self-reports.
   - A judge that gates and re-plans creates selection pressure toward outputs that score well on the rubric rather than outputs that are correct. Detection without ground truth remains open.

8. **Abstention evaluation at the sub-task level in multi-agent pipelines.**
   - AbstentionBench (#46) and Know Your Limits (#45) evaluate single-model, whole-answer abstention.
   - There is no benchmark or metric for partial, sub-task-level abstention inside agent pipelines.
   - Conformal factuality's claim back-off (#41) is the nearest analogue, and #45 lists follow-up actions after abstaining as open.

---

## 2. Evaluation protocol an A* reviewer would now expect for a judge-driven system

1. **Human-agreement study of the judge, per criterion.**
   - Norms: MAST (#24: human–human κ 0.88, judge–human κ 0.77, plus out-of-domain κ); Agent-as-a-Judge (#12: 3 experts, majority consensus); PaperBench/JudgeEval (#14: judge F1 on a held-out meta-evaluation set); JUDGE-BENCH (#2: validate per property).
   - For SciWeaver:
     - 2–3 bioinformatics experts label a few hundred node outputs and final answers.
     - Report human–human κ, judge–human κ, and accuracy for each of the 4 PoT criteria and 6 final criteria.
     - Validate the 4 failure labels with κ and a confusion matrix.

2. **Judge-family swaps and self-preference controls.**
   - Norms: #3, #4, #5; #10 (deduplicate by family).
   - Rerun with judges from at least 2–3 families, including one different from the generator. Report rank stability (Kendall τ) of conclusions and system rankings.

3. **Bias probes.**
   - Norms: MT-Bench (#1) position swaps and verbosity; CALM (#4) perturbations (authority, fake citations, verbosity).
   - Especially needed for the Grounding criterion.

4. **Calibration metrics on the fitness score and the abstention decision.**
   - Metrics: ECE, Brier, AUROC, and risk–coverage / AURC.
   - Norms: #7, #8, #42, #43; TC-ECE and trajectory-level calibration from #44; AECE from #9.
   - Expected: reliability diagrams for tau, and a conformal or fixed-sequence-testing guarantee on a held-out calibration split (#7, #39, #40, #41).

5. **Reference-grounded or objective subsets that bypass the judge.**
   - Norms: No Free Labels (#6), JudgeBench (#5).
   - Use questions with verifiable answers (BixBench-style) and executable code checks, so the headline gains do not rest on the judge alone.

6. **Attribution accuracy on public benchmarks plus your own.**
   - Report agent-level and step-level (node-level) accuracy on Who&When and TRAIL, ideally also Who&When Pro (#23, #25, #33).
   - Add a SciWeaver-specific annotated failure set. Optionally use counterfactual-credit ground truth (#37) or injected faults (#27, #33).

7. **Equal-budget repair baselines with replay control.**
   - Norms: Huang et al. (#18), Repair or Resample (#34).
   - Compare against self-consistency / majority vote (#20), full rerun, Self-Refine, CRITIC (#16, #27) and AgentDebug (#30) at matched tokens.
   - Use prefix replay to separate real repair from resampling.
   - Ablate the failure labels by shuffling them: does label-conditioned repair matter?

8. **Statistical rigour and scale.**
   - N far above 30. Use BixBench (#47) and a larger in-house set.
   - Multiple seeds, confidence intervals, and corrected significance tests (C3 #37 uses Holm correction).
   - Report cost, latency and token use (#16, #18, #29).

9. **External-feedback verification.**
   - Norms: CRITIC (#16), Huang (#18).
   - Tool-grounded checks are expected; purely introspective LLM verification is now a known weak baseline.

---

## 3. Relevant benchmarks

| Benchmark | What it tests | URL |
|---|---|---|
| Who&When | agent/step failure attribution, 184 failures | https://arxiv.org/abs/2505.00212 ; data https://huggingface.co/datasets/Kevin355/Who_and_When ; code https://github.com/ag2ai/Agents_Failure_Attribution (both seen in search results) |
| Who&When Pro | 12,326 traces, multimodal, injected golden labels | https://arxiv.org/abs/2607.09996 |
| MAST / MAST-Data | 14-mode multi-agent failure taxonomy, 1,600+ traces | https://arxiv.org/abs/2503.13657 |
| TRAIL | span-level error localisation, GAIA / SWE-Bench | https://arxiv.org/abs/2505.08638 |
| TracerTraj-2.5K | counterfactual and injected attribution labels | https://arxiv.org/abs/2509.03312 |
| AgentErrorBench | 200 root-cause-annotated trajectories | https://arxiv.org/abs/2509.25370 |
| SymFail (+ SymTrace) | 536 failures; replay-controlled repair evaluation | https://arxiv.org/abs/2608.25920 |
| Aegis-Bench | attribution with error modes (used in AgentLocate) | [unverified; not opened] |
| JudgeBench | objective-label judge accuracy | https://arxiv.org/abs/2410.12784 |
| MT-Bench / Chatbot Arena | judge–human agreement, biases | https://arxiv.org/abs/2306.05685 |
| JUDGE-BENCH | 20 human-annotated tasks | https://arxiv.org/abs/2406.18403 |
| DevAI | requirement-graph agent judging (Agent-as-a-Judge) | https://arxiv.org/abs/2410.10934 |
| AgentRewardBench | LLM judges of web-agent trajectories | https://arxiv.org/abs/2504.08942 |
| PaperBench + JudgeEval | rubric-judged research replication, judge meta-evaluation | https://arxiv.org/abs/2504.01848 |
| BFF-Bench / VERDICTS | expert-grounded judge correctness | https://arxiv.org/abs/2503.05061 |
| AbstentionBench | abstention on unanswerable questions | https://arxiv.org/abs/2506.09038 |
| BixBench | bioinformatics agent analyses, about 300 open-answer questions | https://arxiv.org/abs/2503.00096 |
| ALFWorld / WebShop / BabyAI / TextCraft | PRM / step-verifier evaluation (#21, #22) | via the papers above |

**Repository URLs not opened [unverified]:** the DevAI repository, the TRAIL Hugging Face dataset, the JudgeBench GitHub, the AbstentionBench GitHub, the BixBench Hugging Face dataset, and the PaperBench repository. Use the arXiv links above as canonical references.

**Bottom line for the follow-up.** The defensible A* contribution is a calibrated per-node verifier for DAG pipelines. Concretely:
- A conformal or fixed-sequence-calibrated tau with a composed guarantee, plus validated failure-type labels.
- Attribution that exploits the explicit DAG, checked against exact counterfactual credit, which is cheap at 4 agents.
- Repair evaluated with prefix replay against equal-budget resampling.
- An expert human-agreement study, judge-family swaps, ECE / risk–coverage reporting, and BixBench-scale evaluation.

The closest competitors to position against are AgentMonitor (#26), AgentLocate (#31), Repair or Resample (#34), Trust or Escalate (#7) and KnowNo (#39).