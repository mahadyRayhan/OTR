# Literature review: SciToolAgent, its references, and the work that followed it (revising the D1-D4 plan)

**Scope and method.** SciToolAgent (Ding et al., arXiv 2507.20280; Nature Computational Science 2025) was read in full from arXiv v1, all 21 pages. All 43 of its references were checked:

- The 23 research papers were read in full or by their limitations, discussion and conclusion sections.
- Survey papers were read by their open-problems sections.
- Tool and database papers were skimmed by abstract.
- Birhane et al. was read by abstract only (paywalled). Review [8] was read by abstract and Section 6.

Work published since SciToolAgent (July 2025 to November 2026) was found through Semantic Scholar's list of about 110 citing papers, plus arXiv and web search. Quotes are from the papers. Items marked **[unverified]** come from snippets or secondary sources.

---

## 1. SciToolAgent in brief

- **System.** A Planner, an Executor and a Summarizer, all one LLM (GPT-4o by default), over more than 500 tools in biology, chemistry and materials science.
  - The tools sit in **SciToolKG**, a hand-built graph. Tool nodes link to attribute nodes: purpose, inputs and outputs, category, source, safety level.
  - The Planner retrieves tools by graph search (top-k by similarity, then d-hop neighbours, with d=3, k=5, n=10) and writes a linear "chain of tools".
  - The Executor fixes inputs with "predefined rules or heuristics".
  - A safety check flags molecules or proteins that resemble entries in PubChem and UniProt toxic lists (similarity above 0.95), and only for tools labelled high-risk.
  - The Summarizer can ask the Planner to re-plan.
- **Evaluation.** SciToolEval has 531 questions: 152 single-tool and 379 multi-tool.
  - GPT-4o wrote the questions, and ReAct with GPT-4o produced the answers. Three experts reviewed them.
  - A GPT-4o similarity judge grades the answers.
  - Overall answer accuracy is 94.0%, against ReAct 76.7, Reflexion 84.7, CACTUS 82.9 and ChemCrow 83.4. o1 scores 95.7, Qwen2.5-72B 88.9 and a LoRA-tuned Qwen2.5-7B 85.2.
- **Limitations the authors state (Section 3).**
  - "the manual construction of the SciToolKG knowledge graph … its scalability is constrained by the effort required to curate and update tool information"
  - dependence on proprietary LLMs that "may be financially and technically inaccessible"
  - "even with fine-tuning, Qwen2.5-7B-FT still lags behind GPT-4o, especially in complex tool planning and multi-step reasoning"
- **Limitations they do not state:**
  1. **Circular benchmark and judge.** GPT-4o wrote the questions, produced the reference answers and judges the results.
  2. **No ablations.** Nothing tests the system without the KG, without the safety check, or with different retrieval settings.
  3. **No cost or latency figures.**
  4. **A single linear chain.** One LLM plays three roles, there is no parallelism, and repair is rule-based.
  5. **Uncalibrated safety threshold.** The check has no precision or recall evaluation.
  6. **Figure 2 contains impossible values.** It shows negative tool accuracies for Qwen2.5-7B.
  7. **Unreported settings.** The embedding model, prompts and fine-tuning data size are not given.
  8. **Shared pipeline for benchmark and training data.** The fine-tuning data comes from the same SciToolKG and GPT-4o pipeline as the benchmark.

---

## 2. What the references say (grouped)

### 2.1 Tool learning and agent methods

| Ref | Paper | Stated limitations (quoted) | Use for us |
| --- | --- | --- | --- |
| [2] | Toolformer | "inability … to use tools in a chain"; "does not take into account the tool-dependent, computational cost" | Its loss-reduction filter keeps a tool call only if the result helps. That is a **precedent for D1's utility-by-intervention**: cite it as lineage. D1's difference: black-box agents, effects several hops away, a budget for the whole DAG, and calibration. |
| [3] | GPT4Tools | Success rate "not 100%"; long fixed prompts | A three-part routing metric: the call decision, the tool, and the arguments. Negative "no tool" samples correspond to an abstain action. Shows a small model can be distilled into a router. |
| [21] | ReAct | Reasoning errors 47% of failures; "Non-informative search … 23% of the error cases, derails the model" | The derailment quote states D1's motivation in one line. Its error taxonomy is a template for D2. A mandatory baseline at equal tokens. |
| [23] | ToolNet | "graph construction demands tool-use trajectories, which can be costly"; tested on gpt-3.5 only | A weighted tool-transition graph learned from traces. The **strongest replacement for SciWeaver's regex routing**, and a baseline. Its weights come from uncalibrated LLM scores [-3, 3]; we would use counterfactual credit instead (D2). |
| [24] | ToolkenGPT | Needs model weights | Little use with API models. |
| [25] | Reflexion | "not having a formal guarantee for success"; self-written tests give 16.3% false positives on MBPP; HotpotQA gains use oracle exact-match feedback | The "retry is resampling luck" baseline for D2. Report P(wrong \| check passes). Executable checks need calibration too. |
| [42] | LoRA | Adapters selected by heuristics | Cheap adapters for D1's scorer, D2's verifier and a learned router on an open model. |

### 2.2 Chemistry and materials agents

| Ref | Paper | Key evidence or stated limitation | Use for us |
| --- | --- | --- | --- |
| [11] | ChemCrow | "GPT-4 as an evaluator cannot distinguish between clearly wrong GPT-4 completions and Chemcrow's performance" | **Published precedent for our judge result** (judge correctness vs exact correctness, rank correlation 0.23). It anchors the evaluation section and D2. |
| [12] | CACTUS | 1,000 template questions with exact answers computed by tools; prompt changes swing weak models (Falcon 71.6 → 16.8%) | A **template for a judge-free materials gold set** (pymatgen, Materials Project, ASE). A weak-model axis for sweeps. |
| [18] | ChatMOF | Accuracy reported **excluding** token-limit failures; the genetic algorithm is capped near 100 structures by context | **D1 pull-by-pointer** for large outputs. Count context-overflow failures in the denominator. |
| [17] | LLMatDesign | Without history it needs 26.6 steps, against 10.8 with it; surrogates stand in for DFT | Measured causal value of passed-forward context, a **D1 testbed**. Choosing surrogate, MLFF or DFT is a **D4 fidelity-planning** problem. |
| [20] | MechAgents | "script runs smoothly without errors but the assignment is not necessarily fulfilled"; "self-correction is not sufficient in handling conceptual errors" | A D2 **silent-failure** label and physical-invariant checks. Test a critic agent against a calibrated gate. |
| [19] | ChatEDA | 50 tasks; an executability gate, then blind multi-judge grading; a fine-tuned controller beats GPT-4 | A D2 protocol. A fine-tuned small decomposer is credible. |
| [9] | Coscientist | "labelling is inherently subjective"; GPT-3.5 failed on JSON format | Measurement as the gold standard. Typed atoms with schema checks. |
| [4], [8], [10] | Benchmark and reviews | "chemistry-related tasks necessitate exact matching"; "lack of robust benchmarks and evaluation schemes" | Grade with exact match. Exact strings belong to execution or databases. |

### 2.3 Biology agents and safety

| Ref | Paper | Key evidence or stated limitation | Use for us |
| --- | --- | --- | --- |
| [15] | GenoTEX / GenoAgent | 1,384 gene-trait problems; references from 9 bioinformaticians; human AUROC ceiling 0.89; o1 reaches about 0.74 at about $1.96 and 192 s per problem. "additional rounds yield diminishing returns"; "the Code Reviewer agent often provides inconsistent feedback containing technical errors" | **The strongest testbed.** Executable, gold answers, a human ceiling and a cost figure. Published motivation for D2: self-correction plateaus and the reviewer is unreliable. A real data workload (41.5 GB) for D4. |
| [13] | GeneGPT | GeneTuring: 9 tasks × 50 questions, exact match (0.83 vs ChatGPT 0.12). GeneHop: multi-hop. Error types E1 wrong or no API, E2 wrong arguments, E3 answer not extracted, E4 not answerable | A cheap gold set for bioinformatics. E1–E4 give **gold failure labels for D2**. Needs real NCBI calls. |
| [14] | CRISPR-GPT | Does not let the LLM "dynamically add/delete new tasks"; blocks sequences of 20 bp or more from reaching external LLMs | A D1 **SENSITIVE atom type** (never forwarded). A template vs dynamic DAG trade-off. A completeness–conciseness preference for D3. |
| [16] | ProtAgents | Agents "cannot mitigate" an upstream retrieval error; qualitative only | Errors that cannot be recovered downstream motivate D1 and D2. A broadcast-chat (full-context) baseline. |
| [22] | SciGuard / SciMT | "over-restriction of safe substances or oversight of genuinely hazardous ones" | **Safety belongs inside D2** as a calibrated, asymmetric-risk gate plus an abstain action. Report harmful-pass and over-refusal rates on SciMT biology subsets. |

### 2.4 Surveys and tool papers

- **Zhao et al. [5]**
  - LLM evaluators show "order bias, self-preference bias, and length bias", and "might not be adequate for evaluating … complex tasks". Supports D2.
  - Multi-agent communication cost grows "as the number of agents increases", and the survey calls for "more effective and efficient communication protocols". That is D1's problem, nearly word for word.
  - "data contamination has become a severe issue". Check our public test sets.
- **Min et al. [6]:** "efficiency … as a main evaluation criterion, together with accuracy". Report tokens, dollars and latency everywhere.
- **Wang et al. [7]**
  - "the prompt for one module has the potential to influence others". Supports blame assigned at the DAG level.
  - "generating a flawless plan … is extremely difficult". Supports re-planning.
  - Caution: Wang's "knowledge boundary" means simulation realism, not our PoT label of the same name.
  - None of the three surveys names the personalization of plans as an open problem.
- **Tools with their own confidence scores** are calibrated outside signals a D2 verifier can use. SciToolAgent calls these tools but ignores their confidence.
  - Molecular Transformer [31]: its uncertainty score is "89% accurate in terms of classifying whether a prediction is correct".
  - ESMFold confidence [27].
  - MOFSimplify uncertainty [36].
  - BLAST E-values [38].
- **Fidelity and cost ladder (D4).** Cache, then surrogate, then full computation: MOFX-DB [35], then MOFSimplify [36] or ANM [28], then RASPA [37]. Likewise BLAST [38] against Smith-Waterman [41].
  - SciToolAgent's own MOF case runs RASPA on every MOF that survives the first filter, before applying a cheap price constraint. A cost-aware planner would check the cache, filter cheaply, and simulate last.

---

## 3. Work since SciToolAgent (July 2025 to November 2026)

| Work | What it does | Effect on our plan |
| --- | --- | --- |
| SciToolAgent-Evo, arXiv 2607.28692 (same group) | Adds new tools to SciToolKG automatically and uses a LinUCB bandit to choose between known and new tools. Ships OpenSciToolBench (900 tasks), ablations and cost per task ($0.018 vs $0.007) | Fixes the manual KG, the missing ablations and the missing cost. Still sequential, and its benchmark is generated and graded by an LLM |
| LIBRA, arXiv 2611.05445 (3 Nov 2026) | An LLM proposes tool contracts; sandbox tests keep only the relations that work at runtime. **Evaluated on SciToolEval** (Recall +7.9, NDCG +3.7) | Already covers "execution-verified tool graph" |
| ToolRosella 2603.09290; ToolUniverse 2509.23426; SING 2606.16591; Tool Graph Retriever 2508.05152; DTDR 2512.17052 | Tools built automatically from GitHub; 600+ tools; intention–tool graphs; learned dependency graphs | Tool inventories and graphs built without hand curation |
| Rethinking Tool Graph Memory, 2610.21571 | Over 9 models, tool-graph memory "seldom improves significantly"; random guidance keeps most of the gain | Undercuts any direction built on a tool graph |
| Self-Healing Tool Routing 2603.01548; CITA 2610.02330 | A cost-weighted graph with deterministic repair; a learned value for the next tool call | Related work for D2 repair and for learned routing. CITA is adjacent to D1, but it values actions, not context |
| Triage-PiEvo 2610.14683; DrugHarness 2610.23791; "Plausible but Wrong" 2604.25345 | Staged verification gates; skills admitted only after execution; silent failures documented | New D2 neighbours. None is calibrated or composes guarantees along a DAG |
| Bioinfoysis 2609.03871; K-Dense Analyst 2508.07043 | Hierarchical multi-agent bioinformatics with validated execution | Competition for SciWeaver's system claims. Cite and position against them |
| Hierarchical Server Architecture for Agentic Science 2608.05332; SimulCost 2603.20253 | Agents choose among 51 cloud, edge and HPC providers; a cost-aware simulation benchmark | The closest D4 work. Neither plans the analysis and the resources together |
| Identity or Evidence? 2610.36510; BenchGuard 2604.24955 | LitQA2 and Biomni items answerable from entity identity alone; defects in BixBench and ScienceAgentBench | Report identity-blinded LitQA2 results. Audit any benchmark we adopt |

**Already addressed by others, so not ours to claim:**
- the manual tool KG;
- KG ablations and the KG's value;
- cost reporting for tool agents;
- partly, dependence on proprietary models (SciAgent-8B, Biomni-R0 **[unverified details]**);
- benchmarking weak safety.

**Still open:**
1. Calibrated verification. No conformal or risk-controlled per-node gates on a DAG were found.
2. Orchestration at the DAG level with credit and blame. Evo is still sequential.
3. Gold-answer, judge-free evaluation of full orchestrators.
4. A learned user model that steers decomposition.
5. Planning the analysis and the compute resources together under a cost and time utility.

---

## 4. Revised plan

### Considered and rejected as a new lead: an automatically built, execution-verified tool graph with learned routing

SciToolAgent's main stated limitation (the manual KG) invites this direction, but it is now **high risk**:
- LIBRA (November 2026, evaluated on SciToolEval) covers execution-verified relations.
- SciToolAgent-Evo, from the same group, covers automatic growth and learned selection.
- 2610.21571 reports that tool graphs seldom help.

Two defensible pieces remain, and both fold into existing directions:
- **(a)** Routing over agents, tools and compute learned from outcomes at the DAG level, as part of D1 and D2. This replaces SciWeaver's regex routing.
- **(b)** Optional and cheap: an analysis paper, "Does the tool graph help? KG on vs KG off with gold answers", rerunning SciToolAgent (code is public) on gold-answer tasks.

### Cross-cutting changes, before D1-D4

1. **Make judge-free evaluation an explicit contribution.** The evidence:
   - ChemCrow's judge failure;
   - Zhao's "might not be adequate";
   - SciToolAgent's and OpenSciToolBench's circular evaluation;
   - our own correlation of 0.23.

   Report exact accuracy, plus tokens, dollars and latency.
2. **Shared executable testbed.**

   | Domain | Sets |
   | --- | --- |
   | Bioinformatics | GeneTuring and GeneHop; a GenoTEX subset; LAB-Bench2 SeqQA2 and LitQA3 (with verifiers); Biomni-Eval1; BixBench multiple choice |
   | Materials | A CACTUS-style executable gold set (pymatgen, Materials Project, ASE); ChatMOF questions; MatToolBench **[scoring partly unverified]** |
   | Cost (D4) | SimulCost |

   Plus a contamination check and identity-blinded LitQA2.
3. **A sandboxed executor as shared infrastructure:** NCBI E-utilities, BLAST and a small Python sandbox.
   - GeneTuring, GenoTEX and SeqQA all need it, and D2's executable checks depend on it.
   - It runs short checks only, never heavy jobs such as training. SciWeaver still suggests code to the user.
   - This reverses the earlier "no code run for now" decision, so it needs **your approval**.

### D1. Causal-utility context routing (keep as the lead; novelty risk low to medium)
- **Lineage:** cite Toolformer's loss-reduction filter. State the difference: black-box agents, multi-hop effects, a joint DAG budget, calibration.
- **Atom types:** add SENSITIVE (never forwarded, following CRISPR-GPT's privacy filter) alongside CONSTRAINT (never dropped). Widen CONSTRAINT to cover hazards (PubChem) and input/output format chains.
- **Pull-by-pointer for large artifacts:** count ChatMOF-style context-overflow failures.
- **Testbeds:** LLMatDesign's loop, whose history ablation shows measurable causal value; GeneHop multi-hop; GenoTEX.
- **Budget:** price compute and agent calls, not only tokens.
- **Baselines and related work:** add ToolNet-style routing and CITA. ProtAgents' broadcast chat serves as the full-context baseline. Recheck RCR-Router, MessageSieve and ACON for updates.

### D2. Calibrated, DAG-aware verification and repair (crowded, but its core is still open)
- **Central claim:** "grounded and calibrated, not LLM-judged". Lead with P(wrong | gate passes) and risk–coverage curves.
- **Verifier features:**
  - confidence that tools report about themselves: Molecular Transformer, ESMFold, BLAST E-values, MOFSimplify;
  - database and executable checks;
  - physical-invariant checks.
- **Labels:**
  - a new silent-failure label (MechAgents, "Plausible but Wrong");
  - GeneGPT's E1–E4 as gold failure labels for validating PoT's labels.
- **Motivation:** GenoTEX's plateau after one correction round and its inconsistent reviewer.
- **Safety folded in:**
  - an asymmetric-risk conformal gate plus abstain/escalate (SciGuard);
  - evaluated on SciMT biology subsets for harmful-pass and over-refusal rates;
  - SciWeaver's real risks are patient and genomic data privacy, licences, and dual-use pathogen queries.
- **New baselines:** Reflexion at equal tokens, Triage-PiEvo, DrugHarness, Self-Healing Router, Bioinfoysis, MechAgents' critic.
- **Learned routing weights:** ToolNet-style, updated by counterfactual credit. This links D1 and D2 and replaces regex routing.

### D3. Intent-aware scientific orchestration (still open; novelty risk low to medium)
- The surveys give little support for the gap, so rely on PDR-Bench and PersonaAgent.
- Cite Graph of Trace (ACL 2026), Interactive Deep Research and WebVisus.
- **A measurable preference:** does the intent profile predict where a user sits on CRISPR-GPT's completeness–conciseness trade-off? Reuse their four-criterion rubric, with inter-rater agreement.
- **Expertise-aware verification and safety depth,** following Wang's "users must possess the necessary expertise".

### D4. Compute-aware research planning (the journal; novelty risk medium)
- **Must cite:** Hierarchical Server Architecture (2608.05332) and SimulCost.
- **The gap that remains:** planning the analysis and the resources together under a utility over cost, time and fidelity.
- **Extend beyond nf-core sizing:**
  - plan over a fidelity ladder: cache → surrogate → full simulation;
  - cover CPU-bound work (RASPA) and GPU-bound work (Chroma, ESMFold);
  - use SciToolAgent's own MOF and protein cases as examples where reordering saves money;
  - add the GenoTEX data workload and LLMatDesign's surrogate-vs-DFT choice.
- **Check** the SC, eScience and PEARC workshops before claiming "first".

### For the current SciWeaver journal revision
- **Cite and position against:** SciToolAgent, SciToolAgent-Evo, Biomni, K-Dense Analyst and Bioinfoysis.
- **Cite ChemCrow** next to our judge-vs-truth result.
- **Optionally run SciToolAgent** (public code) on our gold sets as a tool-executing baseline.

### Order
1. Finish the revision with the gold-answer evaluation.
2. Build the sandbox executor and the shared executable testbed. Approval needed.
3. Run D1's December go/no-go test as planned.
4. Then D2.
5. D4 alongside, as the journal paper.
6. D3 when a user study is feasible.

---

## 5. Not verified

- Birhane et al. (abstract only).
- CRISPR-GPT's 288-case Gene-Editing-Bench in the Nature BME version.
- Scoring details for Biomni-Eval1, SciSkillBench, BioXArena and MatToolBench.
- Whether ToolRosetta and ToolRosella are the same work.
- Details of SciAgent-8B, Biomni-R0 and RADAgent.

Most 2026 items come from arXiv abstracts. Read the full text before citing a specific number.
