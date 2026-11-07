# Literature review: LLM agents for scientific research, their benchmarks, and personalization (for the A*-venue SciWeaver follow-up)

**Scope and method.** This covers 44 papers. For each one I opened the arXiv abstract plus the full text (arXiv HTML, the PDF via pdftotext, or the PMC page for Biomni and Coscientist), and pulled limitations and future work from the paper's own sections. Venues come from the arXiv comment or journal-ref field, or from a proceedings or press page found in search.

- **[venue unverified]** marks a venue I could not confirm. DBLP and the Semantic Scholar API were rate-limited, and nature.com blocked automated access.
- **"None stated"** means I found no dedicated limitations or future-work text in the version I opened.

---

## A. End-to-end and "AI scientist" systems

**1. Coscientist.** Boiko et al., *Nature* 2023 (arXiv preprint 2304.05332, "Emergent autonomous scientific research capabilities…"). https://pmc.ncbi.nlm.nih.gov/articles/PMC10733136/
- **Core idea:** A GPT-4 planner with web search, documentation search, code execution and lab-automation APIs (Opentrons, Emerald Cloud Lab).
- **Headline:** It designed and ran Pd-catalysed Suzuki and Sonogashira couplings, with products confirmed by GC-MS. It always found the right cloud-lab functions and wrote HPLC code that ran. Its GPT-4 optimizer beat Bayesian optimization on normalized advantage over Suzuki and Buchwald–Hartwig lookup datasets.
- **Limitations:** Plates were moved by hand, so it was not fully automated. An HPLC air bubble shows the need for automated QC. GPT-3.5 often broke the JSON schema. Grading synthesis quality is subjective. Code and prompts were not fully released for dual-use reasons.
- **Future work:** Add Reaxys and SciFinder to the action space. Try ReAct, chain-of-thought and tree-of-thoughts. Use web search to set follow-up experiment parameters.
- **Relevance:** Its **documentation-search module** (retrieving API docs to ground code) is exactly the "tool documentation" knowledge source in SciWeaver's future work. Coscientist acts; SciWeaver only recommends.

**2. ChemCrow.** Bran et al., *Nature Machine Intelligence* 2024 [venue unverified; arXiv v5 only says "Experimental results"]. https://arxiv.org/abs/2304.05376
- **Core idea:** GPT-4 with 18 expert-designed chemistry tools on LangChain.
- **Headline:** Planned and executed syntheses of an insect repellent and three organocatalysts, and guided discovery of a new chromophore. **GPT-4 acting as evaluator could not tell clearly wrong GPT-4 answers from ChemCrow's**, while human experts preferred ChemCrow.
- **Limitations:** Hallucination and faulty logic persist; tools cannot fix flawed reasoning. LLM-based evaluation favours fluent, verbose answers, forcing reliance on slow human evaluation. Closed APIs are not reproducible. Task selection has implicit bias. Performance is bounded by the quality of the tools.
- **Future work:** Language and image tools, broader evaluation tasks, better chemistry-ML tools and agent architectures.
- **Relevance:** This is direct evidence against SciWeaver's **LLM-as-judge-only evaluation**. Reviewers will cite it. Add expert ratings and a judge–human agreement study.

**3. The AI Scientist (v1).** Lu et al., arXiv 2024. https://arxiv.org/abs/2408.06292
- **Core idea:** Fully automated ML research loop: idea, code (via Aider), experiments, LaTeX paper, automated review.
- **Headline:** Under $15 per paper. The automated reviewer is near-human. Some papers pass the acceptance threshold according to that reviewer.
- **Limitations:**
  - Ideas are very similar across runs.
  - Aider fails to implement many ideas, and implementations can be wrong without anyone noticing.
  - Too few experiments for rigour, which leads to deceptive conclusions.
  - No vision, so it cannot read its own figures.
  - Poor citation finding.
  - Numeric comparison errors.
  - The reviewer may be contaminated by ICLR 2022 data and cannot run a rebuttal.
- **Future work:** Vision, follow-ups on its best ideas, recent papers as a novelty source, an adversarial code-checking reviewer.
- **Relevance:** A fully autonomous paradigm; SciWeaver is an assistive, human-facing copilot. Use it to position SciWeaver.

**3b. "Towards End-to-End Automation of AI Research"** (the AI Scientist journal paper). Yamada/Lu et al., *Nature* 651:914–919 (2026). https://arxiv.org/abs/2606.15497
- **Headline:** Template and template-free modes. One of three manuscripts passed first-round review at an ICLR 2025 workshop (70% acceptance rate).
- **Limitations:** Not top-tier, not even consistently workshop-level. Naive ideas, wrong implementations, weak rigour, duplicated figures, hallucinated citations. Computational experiments only.
- **Future work:** Apply the playbook to other domains where experiments can be automated.

**4. The AI Scientist-v2.** Yamada et al., arXiv 2025. https://arxiv.org/abs/2504.08066
- **Core idea:** Template-free agentic tree search with an experiment-manager agent and a vision-language-model reviewer.
- **Headline:** First fully AI-generated paper to pass peer review, at an ICLR workshop (1 of 3 accepted).
- **Limitations:** Workshop-level only and inconsistent. Struggles with genuinely novel hypotheses, innovative methodology and justifying design choices with deep domain expertise. The community needs norms on disclosure of AI-generated science.
- **Future work:** None specific beyond scaling with better LLMs.
- **Relevance:** Its tree search plus experiment manager is a richer re-planning scheme than SciWeaver's single judge-triggered re-plan.

**5. Agent Laboratory.** Schmidgall et al., Findings of EMNLP 2025 (confirmed via ACL Anthology author page). https://arxiv.org/abs/2501.04227
- **Core idea:** A human-idea-driven pipeline (literature review, experimentation, report writing) with autonomous and **co-pilot** modes.
- **Headline:** o1-preview gives the best outcomes. Human feedback at each stage improves quality. 84% cost reduction versus prior autonomous methods. **Automated reviewer scores do not predict human reviewer scores.**
- **Limitations:**
  - Self-evaluation through emulated NeurIPS reviewers is unreliable.
  - Rigid paper structure and a two-figure cap.
  - No repository-level code management.
  - Hallucinated experimental results.
  - Instruction-following failures in the literature-review phase (looping on "summarize").
  - Retrieved papers overflow the token limit.
- **Future work:** Flexible repo-level editing, more figures.
- **Relevance:** Closest in philosophy (a co-pilot that keeps the human in the loop). It also shows again that LLM judges diverge from humans.

**6. AI co-scientist (Google).** Gottweis et al., arXiv 2025; the v2 arXiv comment says "*Nature* (2026)". https://arxiv.org/abs/2502.18864
- **Core idea:** Gemini multi-agent system (Generation, Reflection, Ranking via Elo tournament, Proximity, Evolution, Meta-review) with test-time compute scaling.
- **Headline:** Drug-repurposing candidates for acute myeloid leukaemia validated in vitro. Liver-fibrosis targets. Recapitulated an antimicrobial-resistance mechanism.
- **Limitations:**
  - Open-access literature only, so it misses prior work and has **no negative-results data**.
  - Weak multimodal reasoning (figures, multi-omics, knowledge graphs).
  - Inherits LLM hallucination.
  - Elo is a limited, self-favouring metric.
  - Validation is preliminary and does not cover drug delivery or pharmacokinetics.
- **Future work:** Better literature review and citation recall, factuality cross-checks with tools, larger multi-expert evaluations across disciplines, RL for ranking, integration of public databases, datasets and images, lab-automation closed loops, structured feedback interfaces.
- **Relevance (important):** Its Meta-review agent does **"Research contacts identification"**: it suggests qualified domain experts, with reasoning, for review and collaboration. That **overlaps with SciWeaver's Scholar Recommender**, so SciWeaver can no longer claim collaborator recommendation is unique. Its expert-in-the-loop design is also a model for SciWeaver's re-planning.

**7. Biomni.** Huang et al., bioRxiv 2025 (DOI 10.1101/2025.05.30.656746). Secondary coverage says it was later published in *Science* [unverified]. https://pmc.ncbi.nlm.nih.gov/articles/PMC12157518/
- **Core idea:** Biomni-E1 is an environment of 150 tools, 105 software packages and 59 databases, mined by an LLM from recent bioRxiv papers. Biomni-A1 is a code-as-action agent that uses it.
- **Headline:**

| Benchmark | Biomni | Comparison |
|---|---|---|
| LAB-Bench DbQA | 74.4% | human 74.7% |
| LAB-Bench SeqQA | 81.9% | human 78.8% |
| Humanity's Last Exam, biomed subset | 17.3% | base LLM 6.0% |

  Wet-lab-validated cloning protocol. A vendor blog reports Biomni Lab at 52.2% on BixBench in Feb 2026 [unverified vendor claim].
- **Limitations:** Covers only a subset of the field. Struggles with nuanced clinical judgment, novel experimental reasoning and deep biological synthesis. Not expert-level everywhere. Mining recent literature may miss foundational methods.
- **Future work:** RL training, multimodal inputs, automatic discovery of new tools and databases (including older methods).
- **Relevance:** The strongest bioinformatics generalist. It **covers tools, databases and code but not people or compute**. SciWeaver should cite it, and possibly use Biomni-E1 as its tool and dataset knowledge source.

**8. BioDiscoveryAgent.** Roohani et al., ICLR 2025 [venue unverified]. https://arxiv.org/abs/2405.17631
- **Core idea:** LLM agent for closed-loop design of genetic-perturbation experiments, with tools for literature search, code execution and an agent critic.
- **Headline:** 21% improvement over Bayesian-optimization baselines across 6 datasets and 46% on non-essential genes. More than twice random on gene combinations.
- **Limitations:** Performance varies by cell type with literature coverage. Gains come mostly in early rounds. Weak extraction from non-text sources.
- **Future work:** Better tools and fine-tuning for non-text information.
- **Relevance:** Shows literature coverage drives cross-context generalization, which matters for SciWeaver's cross-domain plan.

**9. CellAgent.** Xiao et al., arXiv 2024 [venue unverified]. https://arxiv.org/abs/2407.09811
- **Core idea:** Planner, Executor and Evaluator roles with hierarchical decision-making and self-iterative optimization for scRNA-seq analysis. Uses GPT-4V to judge results.
- **Headline:** 92% task completion, about twice GPT-4 alone. Batch-correction overall score 0.684 versus 0.642 for scVI.
- **Limitations:** Self-evaluation through GPT-4/4V is imperfect and supports only limited optimization directions. Users with specific preferences must specify the evaluation manually.
- **Future work:** Automatic integration of new tools.
- **Relevance:** Same planner and evaluator pattern as SciWeaver's DAG plus judge. Its stated need for user-preference-aware evaluation **motivates SciWeaver's intent profiler**.

**10. BioPlanner / BioProt.** O'Donoghue et al., EMNLP 2023. https://arxiv.org/abs/2310.10632
- **Core idea:** Protocols converted to pseudocode give automatic evaluation of protocol planning.
- **Headline:** **GPT-4 as evaluator is only slightly above chance** at telling ground truth from generated protocols. One LLM-generated protocol ran successfully in the wet lab.
- **Limitations:** Paid API (about $1,000). Biology only. Misuse risk.
- **Future work:** Open models; extend to chemistry and materials.
- **Relevance:** A second piece of evidence that LLM judges are weak, and its pseudocode-style evaluation is a template for scoring SciWeaver's notebook and workflow recommendations.

**11. CRISPR-GPT.** Qu et al., *Nature Biomedical Engineering* 2025 (per arXiv comment). https://arxiv.org/abs/2404.18021
- **Core idea:** LLM planner plus a state-machine task executor with Meta, Auto and Q&A modes, for gene-editing design.
- **Headline:** Beats general LLMs in expert evaluation. Wet-lab validation.
- **Limitations:** Cannot generate full constructs or vectors from natural language.
- **Future work:** Integrate FragMID-style modular design; integrate with lab robotics.
- **Relevance:** Its **multiple interaction modes for different user expertise** resemble SciWeaver's Procedural, Analytical and Conceptual intents. It is a good related-work anchor.

**12. GeneAgent.** Wang et al., *Nature Methods* 2025 [venue unverified; nature.com blocked, read the arXiv PDF]. https://arxiv.org/abs/2405.16205
- **Core idea:** Self-verification against 18 curated domain databases for gene-set function naming.
- **Headline:** On 1,106 gene sets, ROUGE-L on MSigDB rose from 23.9 to 31.0. Mean semantic similarity 0.705–0.761.
- **Limitations:** GPT-4 backbone only. Still produces names far from the references. No gene-set preprocessing.
- **Future work:** More domain databases for verification.
- **Relevance:** **Database-grounded verification**, rather than LLM-self-judging, is a concrete upgrade for SciWeaver's judge.

**13. Robin.** Ghareeb et al., arXiv 2025. https://arxiv.org/abs/2505.13400
- **Core idea:** Literature agents (Crow, Falcon) plus a data-analysis agent (Finch) in a lab-in-the-loop.
- **Headline:** Proposed ripasudil for dry age-related macular degeneration and validated it. RNA-seq analysis implicated ABCA1.
- **Limitations:** Produces experiment outlines, not executable protocols. Finch depends on expert prompt engineering. The LLM-judged tournament may not match human judgment.
- **Future work:** Detailed executable protocols, self-adapting prompts per data modality, alignment with scientist judgment.

**14. Kosmos.** Mitchener et al., arXiv Nov 2025. https://arxiv.org/abs/2511.02824
- **Core idea:** A structured world model shared by data-analysis and literature agents. Runs up to 12 h and 200 rollouts, about 42k lines of code and 1,500 papers per run.
- **Headline:** 79.4% of report statements accurate (85% for data-analysis statements, **57% for interpretation statements**). Collaborators said a 20-cycle run equals about 6 months of their own work.
- **Limitations:**
  - Invents obscure metrics and confuses statistical significance with scientific value.
  - No automated way to judge claim novelty or significance.
  - Data limited to about 5 GB; weak on raw images and sequencing files.
  - Cannot fetch external public data.
  - Stochastic, and sensitive to how the objective is phrased.
  - **No mid-run human interaction.**
- **Future work:** Training for "scientific taste."
- **Relevance:** The lack of human steering and phrasing sensitivity are exactly what SciWeaver's intent profiler and re-planning target. Use this as motivation.

**15. STELLA.** Jin et al., arXiv 2025. https://arxiv.org/abs/2507.02004
- **Core idea:** Self-evolving agent with a Template Library and a "Tool Ocean" grown by a tool-creation agent.
- **Headline:** About 26% on Humanity's Last Exam biomedicine, 54% on LAB-Bench DBQA, 63% on LAB-Bench LitQA. Accuracy roughly doubles with more trials.
- **Limitations:** A gap between benchmark results and real lab use (stated only briefly).
- **Future work:** Real-world deployment, human collaboration.
- **Relevance:** A template for SciWeaver's "continual learning" future work.

**16. ToolUniverse.** Gao et al., arXiv 2025. https://arxiv.org/abs/2509.23426
- **Core idea:** A standard for AI–tool interaction covering more than 2,700 scientific tools and over 130 skills. Includes tool refinement, generation and workflow composition.
- **Headline:** Case studies in target assessment, chemistry and clinical-trial safety.
- **Limitations:** It "orchestrates rather than certifies". Conclusions are bounded by the underlying databases and models. ML outputs are estimates. Domain judgment is still needed. The scientist stays in the loop.
- **Future work:** None stated.
- **Relevance:** SciWeaver could expose its four agents as ToolUniverse- or MCP-compatible tools, which makes adding agents (statistics, visualization, database querying) cheap.

**17. Paper2Agent.** Miao et al., arXiv 2025. https://arxiv.org/abs/2509.06917
- **Core idea:** Automatically turns a paper plus its codebase into an MCP server agent.
- **Headline:** AlphaGenome, Scanpy and TISSUE agents reproduce the papers' results. Found an ADHD-associated splicing variant.
- **Limitations:** Fails when the original code is incomplete or poorly documented.
- **Future work:** Extend to data resources and discovery papers.
- **Relevance:** Directly relevant to SciWeaver's **Notebook agent**. Go beyond retrieving notebooks to making them executable through MCP.

## B. Literature, synthesis and ideation agents

**18. PaperQA2.** Skarlinski et al., arXiv 2024. https://arxiv.org/abs/2409.13740
- **Core idea:** Agentic RAG over full texts, with citation traversal. Introduces LitQA2 (248 multiple-choice questions).
- **Headline:**

| | Precision | Accuracy |
|---|---|---|
| PaperQA2 on LitQA2 | 85.2% | 66.0% |
| PhD-level humans | 73.8% | 67.7% |

  WikiCrow had fewer cited-but-unsupported statements than Wikipedia (13.5% vs 24.9%). About 70% of detected contradictions were validated by experts. Cost $1–3 per query.
- **Limitations:** None stated as a section. It notes cost, and that LitQA2 questions leaked into Google-indexed sites.
- **Future work:** Scale literature agents.
- **Relevance:** LitQA2 is a ready benchmark for SciWeaver's Knowledge Discovery agent.

**19. OpenScholar.** Asai et al., *Nature* (Feb 2026; per UW press release, not opened on nature.com). https://arxiv.org/abs/2411.14199
- **Core idea:** RAG over 45M open-access papers, with a trained retriever and reranker and a self-feedback loop. Introduces ScholarQABench (2,967 queries, 208 long-form answers; CS, physics, neuroscience, biomedicine).
- **Headline:** The 8B model beats GPT-4o by 5% and PaperQA2 by 7% in correctness. GPT-4o fabricates 78–90% of citations. Experts preferred OpenScholar-GPT4o over human answers 70% of the time.
- **Limitations:**
  - Small expert-answer set (about 110).
  - Heuristic rubric weights that can be gamed stylistically.
  - Sentence-level citation metrics are too strict.
  - Benchmark is time-stamped.
  - No social science.
  - Does not always retrieve the most representative papers.
  - The 8B model has factual errors.
  - No license-protected papers.
  - Human raters focused on writing quality over correctness.
- **Future work:** Use citation networks and recency metadata in retrieval; better 8B training.
- **Relevance:** A baseline and benchmark for SciWeaver's literature agent. Its suggestion to use citation networks is something the Scholar agent's Semantic Scholar graph already offers.

**20. STORM.** Shao et al., NAACL 2024. https://arxiv.org/abs/2402.14207
- **Core idea:** Perspective-guided question asking produces an outline, then a grounded Wikipedia-style article.
- **Headline:** +25% organization, +10% coverage versus an outline-RAG baseline.
- **Limitations:** Lags human articles on neutrality and verifiability. Source bias toward dominant internet sources. Text only. About 15% of sentences unsupported, mostly through improper inferential linking.
- **Future work:** Multimodal grounded articles.

**21. Co-STORM.** Jiang et al., EMNLP 2024. https://arxiv.org/abs/2408.15232
- **Core idea:** The user observes and steers a roundtable of LLM agents. Includes a dynamic mind map and the WildSeek dataset.
- **Headline:** 70% of users prefer it to a search engine; 78% to a RAG chatbot.
- **Limitations:** **Should tailor the discourse to the user's prior knowledge (skip basics for experts, scaffold novices).** Users want more control. Monolingual. Higher latency.
- **Future work:** Implied by the limitations above.
- **Relevance:** Its first limitation is **exactly SciWeaver's intent and expertise profiling**. Cite it as an explicitly stated open problem that SciWeaver addresses.

**22. ResearchAgent.** Baek et al., NAACL 2025. https://arxiv.org/abs/2404.07738
- **Core idea:** Idea generation over an academic citation graph plus an entity-centric knowledge store, refined by human-preference-aligned reviewing agents.
- **Headline:** More creative, valid and clear ideas, by human and model evaluation.
- **Limitations:** Entity store built from titles and abstracts only, about 3 entities per paper. Hallucination. Reviewer criteria not exhaustive across domains. Unsuited to theoretical sciences.
- **Future work:** Scale and refresh the knowledge store; domain-customized criteria.
- **Relevance:** Its **entity-centric store** is a concrete design for SciWeaver's planned cross-domain entity recognition.

## C. Benchmarks for science agents

**23. MLAgentBench.** Huang et al., ICML 2024 (PMLR v235). https://arxiv.org/abs/2310.03302
- **Setup:** 13 ML experimentation tasks.
- **Headline:** Best agent (Claude 3 Opus) has a 37.5% average success rate, ranging from 100% to 0% depending on dataset recency.
- **Limitations:** None stated as a section.
- **Future work:** More robust agent, harder tasks, human–AI collaboration user studies.

**24. ScienceAgentBench.** Chen et al., ICLR 2025. https://arxiv.org/abs/2410.05080
- **Setup:** 102 tasks from 44 papers in 4 disciplines, including bioinformatics.
- **Headline:** Best agent 32.4% (34.3% with expert knowledge); o1-preview 42.2%.
- **Limitations:** Code generation only. CodeBERTScore and GPT-4o figure-judging are imperfect.
- **Future work:** Rigorous per-capability assessment (literature, ideas, planning); better metrics.

**25. DiscoveryBench.** Majumder et al., ICLR 2025 (proceedings and poster page). https://arxiv.org/abs/2407.01725
- **Setup:** 264 real plus 903 synthetic tasks, 6 domains.
- **Headline:** Best system 25%.
- **Limitations:** Excludes forecasting and simulation tasks, so it skews toward social science. No massive datasets (e.g., TCGA). No multimodal or geospatial pipelines.
- **Future work:** Include natural and physical sciences.

**26. BixBench.** Mitchener et al., arXiv 2025 [venue unverified]. https://arxiv.org/abs/2503.00096
- **Setup:** Jupyter-notebook "capsules". The abstract says over 50 scenarios and nearly 300 questions; the v3 HTML says 61 scenarios and 205 questions.
- **Headline:** Frontier models reach about 17% open-answer accuracy; multiple-choice is near random.
- **Limitations and future work:** Missing many workflows and data types. No human baseline. Reasoning models not yet evaluated.
- **Relevance:** **Notebook-centric bioinformatics**, the most direct fit for SciWeaver's Notebook agent.

**27. BixBench3.** Koch et al., arXiv Aug 2026. https://arxiv.org/abs/2608.25286
- **Setup:** 20 study-scale tasks with 138 artifacts, graded programmatically against published artifacts.
- **Headline:**
  - Scores range from 0.00 (Gemini 3.1 Flash Lite) to 0.48 (GPT 5.6 Sol).
  - 0.36 on datasets under 100 GB versus 0.10 above 100 GB.
  - Average 6.8 h, 102M tokens and $43 per task; worst case $525.
- **Limitations:** Methods must be prescribed. Does not test choosing which analyses are worth doing. Inherits errors in the source papers.
- **Relevance:** Dataset size drives both cost and failure, which is **direct motivation for SciWeaver's Cloud agent**: right-sizing compute is part of success.

**28. LAB-Bench.** Laurent et al., arXiv 2024 (submitted to NeurIPS D&B) [venue unverified]. https://arxiv.org/abs/2407.10362. Public subset: https://huggingface.co/datasets/futurehouse/lab-bench
- **Setup:** More than 2,400 multiple-choice questions on literature, figures, databases (DbQA), sequences and protocols.
- **Headline:** Humans generally beat models. Models refuse often on lookup tasks.
- **Limitations:** Not comprehensive for biology. Partial human coverage.
- **Future work:** Community-driven expansion.

**29. BioCoder.** Tang et al., arXiv 2023; reportedly *Bioinformatics* (ISMB 2024) [venue unverified]. https://arxiv.org/abs/2308.16458
- **Setup:** 1,026 Python and 1,243 Java functions, 253 Rosalind problems, fuzz testing.
- **Headline:** GPT-3.5/4 around 50% versus up to 25% for smaller models.
- **Limitations:** Possible contamination from closed models. GPT-assisted curation bias. Reproducibility.
- **Future work:** Multi-step workflows (e.g., count matrix to cell types).

**30. SciCode.** Tian et al., NeurIPS 2024 D&B. https://arxiv.org/abs/2407.13168 (site: scicode-bench.github.io)
- **Setup:** 80 main problems and 338 subproblems across 16 fields, including materials science.
- **Headline:** Best model 4.6% in the most realistic setting (camera-ready o1-preview 7.7% per search).
- **Limitations:** None stated.

**31. DSBench.** Jing et al., reportedly ICLR 2025 [venue unverified]. https://arxiv.org/abs/2409.07703
- **Setup:** 466 analysis plus 74 modelling tasks.
- **Headline:** Best agent 34.12% on analysis; 34.74% relative performance gap on modelling.
- **Limitations:** None stated beyond calling for better agents.

**32. PaperBench.** Starace et al., ICML 2025 [venue partly verified via a conference listing]. https://arxiv.org/abs/2504.01848 (code: github.com/openai/preparedness)
- **Setup:** 20 ICML 2024 papers, 8,316 rubric leaf nodes, LLM judge.
- **Headline:** Best agent replication score 21.0%. Not above ML PhDs.
- **Limitations:** Only 20 papers. Contamination risk. Rubric creation takes several expert-days each. **The LLM judge is below expert accuracy and non-deterministic.** About $8k per evaluation run.
- **Future work:** Model-assisted rubric creation; adversarial stress tests of judges.
- **Relevance:** Its **JudgeEval** (a benchmark for the judge itself) is the methodology SciWeaver should copy to validate its LLM judge.

**33. MLE-bench.** Chan et al., ICLR 2025. https://arxiv.org/abs/2410.07095 (code: github.com/openai/mle-bench)
- **Setup:** 75 Kaggle competitions.
- **Headline:** o1-preview with AIDE reaches at least bronze in 16.9%.
- **Limitations:** Contamination and plagiarism. Clean, well-specified tasks unlike real R&D. Splits and grading re-implemented.
- **Future work:** Refresh with new competitions.

**34. AstaBench (Ai2).** Bragg et al., ICLR 2026 (per arXiv comment). https://arxiv.org/abs/2510.21652
- **Setup:** More than 2,400 problems spanning literature understanding, code and execution, data analysis and end-to-end discovery. Reproducible production-grade search tools. Cost-aware leaderboard. 57 agents in 22 classes.
- **Headline:**
  - Best overall is Asta v0 at 53.0%; best fully open setup 11.1%.
  - ReAct with gpt-5-mini scores 32% at $0.04 per problem.
  - Literature QA is roughly solved (about 80–85%); coding, data analysis and discovery remain unsolved.
  - Specialized workflows sometimes do worse with gpt-5 than plain ReAct.
- **Limitations and future work:** Refine LLM-as-judge grading; contamination-resistant fresh problems; **benchmarks for human collaboration**; deeper biomedicine coverage.
- **Relevance:** The **single best public replacement** for SciWeaver's 30-query set across literature, data and code. Its cost accounting matches the Cloud agent's cost focus. The finding that specialized workflows lose to ReAct is a threat SciWeaver must test against with a ReAct baseline.

**35. SciArena.** Zhao et al., NeurIPS 2025 D&B (Spotlight). https://arxiv.org/abs/2507.01001
- **Setup:** Arena-style community voting on literature-grounded long-form answers. 47 models, more than 20k researcher votes. SciArena-Eval meta-evaluates LLM judges.
- **Headline:** The benchmark shows automated judges struggle.
- **Limitations:** Excludes older models. Cannot integrate Deep Research products (usage caps, no API).
- **Relevance:** Use SciArena-Eval to calibrate SciWeaver's judge.

## D. Deep-research agents and benchmarks

**36. BrowseComp.** Wei et al. (OpenAI), arXiv 2025. https://arxiv.org/abs/2504.12516 (github.com/openai/simple-evals)
- **Setup:** 1,266 hard-to-find, short-answer questions.
- **Headline:**

| System | Accuracy |
|---|---|
| GPT-4o | 0.6% |
| GPT-4o with browsing | 1.9% |
| o1 | 9.9% |
| OpenAI Deep Research | 51.5% (calibration error 91%) |
| Human trainers | 29.2% solved within 2 h |

- **Limitations:** Sidesteps the real query distribution, long answers and ambiguity.
- **Future work:** Multimodal and interactive browsing.

**37. DeepResearch Bench.** Du et al., arXiv 2025. https://arxiv.org/abs/2506.11763 (github.com/Ayanami0730/deep_research_bench)
- **Setup:** 100 PhD-level tasks across 22 fields. RACE scores report quality; FACT scores citations.
- **Headline:** Gemini-2.5-Pro Deep Research leads RACE (48.88) with 111.21 effective citations per report. Perplexity has better citation accuracy. RACE pairwise agreement exceeds inter-human agreement.
- **Limitations:** Small scale. Domain-curation bias. Human evaluation throughput (600 reports, 30–60 minutes each).
- **Relevance:** RACE and FACT are adoptable for scoring SciWeaver's synthesized reports and citations.

**38. Deep Research Agents: A Systematic Examination and Roadmap.** Huang et al., arXiv 2025 (survey; abstract only opened). https://arxiv.org/abs/2506.18096
- **Content:** Taxonomy (static vs dynamic workflows, single vs multi-agent, MCP tool integration).
- **Stated benchmark gaps:** Restricted access to external knowledge, sequential-execution inefficiency, metrics misaligned with practical goals.
- **Relevance:** Use its taxonomy to place SciWeaver: a dynamic, multi-agent, DAG-parallel system.

## E. HPC and cloud agents (for the Cloud Solution agent)

**39. Incisor.** Laurenzano et al., arXiv Apr 2026. https://arxiv.org/abs/2604.24464
- **Core idea:** Program analysis plus LLM reasoning picks AWS EC2 instances "ex ante" from the executable, inputs and command alone, with runtime-verified fallback.
- **Headline:** 100% first-run success on 31 HPC workloads with frontier coding LLMs. Runtime −54% and cost −44% versus expert constraints plus SkyPilot.
- **Limitations and future work:**
  - Single-node only; multi-node and MPI next.
  - Per-stage constraints for **DAG workflows**.
  - Containers.
  - Julia, R and MATLAB support.
  - Co-optimizing the invocation.
  - Using documentation and web search.
  - **Exposing rationale and accepting natural-language user feedback.**
  - User-specified utility functions over cost and runtime.
- **Relevance:** **The most direct competitor and baseline for SciWeaver's Cloud agent.** Its future work (DAG workflows, user feedback, user utility functions) is precisely where SciWeaver's intent-profiled, DAG-orchestrated design can claim novelty. Reuse its 31-workload set or protocol and its metrics (first-run success, cost, runtime).

**40. Hierarchical Server Architecture for Agentic Science.** Sochat et al., arXiv Aug 2026. https://arxiv.org/abs/2608.05332
- **Core idea:** "Secretary agents" discover, negotiate with and select among 51 real and simulated providers (HPC, cloud, edge) through MCP. Supports the US DOE Genesis Mission.
- **Headline:** 87.71% negotiation accuracy over 19,973 negotiations and 6,952 selections. Selection cost comparable to traditional strategies.
- **Limitations:**
  - Simulated archetypes cannot reflect every environment; no multi-tenancy, security or scheduler policies.
  - False "compatible" verdicts (N=1,611), mostly from confusing software and dependency versions.
  - Agents fail to treat container technologies as interchangeable.
  - Older models (Gemini 2.5).
- **Future work:** Richer archetypes, a scheduling component, temporal constraints.
- **Relevance:** Its lesson that **tool interfaces for agents must be more specific than human command output** is directly applicable to SciWeaver's cloud data collection, including the materials-science data.

**41. Connecting LLM Agent to HPC Resource (Parsl + LangGraph).** Ma et al., arXiv 2025. https://arxiv.org/abs/2502.12280
- **Core idea:** Tool calls dispatched to Parsl workers on the Polaris supercomputer (ALCF) for molecular dynamics.
- **Limitations:** Agent tool-call limits. Login and compute nodes differ (no internet on compute nodes). Long queues.
- **Relevance:** A path from *recommending* compute to *executing* on it.

**42. Evaluating LLMs for HPC workload mapping and scheduling.** Sharma et al., 2025 (*Research in Academic Engineering Journal*; weak venue). https://arxiv.org/abs/2511.11612
- **Setup:** 21 LLMs on one scheduling instance.
- **Headline:** 3 found the optimum, 12 near-optimal, 6 suboptimal. Errors in units, transfer delays and dependencies.
- **Future work:** Explicit constraint validation.
- **Relevance:** Weak evidence (a single instance), but it supports pairing the LLM with a solver or validator for cost estimates.

## F. Personalization and user modelling

**43. LaMP.** Salemi et al., arXiv 2023; reportedly ACL 2024 [venue unverified]. https://arxiv.org/abs/2304.11406
- **Setup:** 7 personalized tasks.
- **Headline:** Retrieval from the user profile gives +12.2% zero-shot and +23.5% fine-tuned.
- **Limitations:** Unrealistic task framing (e.g., citation identification as binary choice). Pretraining leakage. Hard to evaluate long-form personalized generation. Privacy.

**44. PersonaRAG.** Zerhoudi & Granitzer, SIGIR 2024 IR-RAG workshop. https://arxiv.org/abs/2407.09394
- **Core idea:** User-interaction agents plus "cognitive dynamic adaptation" in RAG.
- **Headline:** More than 5% accuracy gain on QA; more than 10% for some LLMs.
- **Limitations:** Many LLM calls, latency and cost, long inputs.
- **Future work:** Fewer calls, concise user-profile representations, more user-centric agents.
- **Relevance:** The closest architecture to SciWeaver's profiler-conditioned RAG. It is workshop-level, so the space is open.

**45. PersonaAgent.** Zhang et al., ACL 2026 (per arXiv comment; search says Findings). https://arxiv.org/abs/2506.06254
- **Core idea:** Episodic plus semantic memory, with the persona (a per-user system prompt) optimized at test time through textual feedback.
- **Headline:** LaMP-1 accuracy 0.919 versus 0.862 for MemBank and 0.837 for ReAct.
- **Limitations:** Textual feedback only (misses implicit or multimodal signals). Privacy risk.
- **Future work:** Federated or privacy-preserving personalization.
- **Relevance:** SciWeaver's Bloom profile is static, per query. PersonaAgent shows how to make it **persistent and test-time-adapted across sessions**, which fits the "adaptive intent profiling" future work.

**46. PDR-Bench (Towards Personalized Deep Research).** Liang et al., ICLR 2026. https://arxiv.org/abs/2509.25106
- **Setup:** 50 tasks across 10 domains × 25 real user profiles = 250 queries. PQR framework: Personalization alignment, content Quality, factual Reliability.
- **Headline:** Open-source agents personalize best (OAgents 6.64) but have low factual accuracy (3.77). Agents struggle to extract preferences from implicit context. Explicit personas help. Memory systems (Mem0, MemoryOS, O-Mem) were tested to bridge the gap.
- **Limitations:** Personas annotated in Chinese. Only a subset of queries run.
- **Future work:** Cross-lingual personas, full-scale runs.
- **Relevance:** **The benchmark to evaluate SciWeaver's intent profiler on.** It also shows the open problem, implicit preferences to explicit profile, that SciWeaver's Bloom profiler addresses.

---

## 1. Gaps: open problems as of late 2026

1. **Reliable evaluation of open-ended scientific output.**
   - LLM judges fail in ChemCrow (cannot spot wrong answers), BioPlanner (near chance) and Agent Laboratory (reviewer scores do not predict humans).
   - PaperBench's judge is below expert level; SciArena-Eval shows judges struggle.
   - Kosmos has no automatic test of novelty or significance.
   - Judge calibration against humans is still unsolved, and SciWeaver's evaluation currently depends on exactly that.
2. **Personalization to user expertise and intent in scientific agents.**
   - Co-STORM explicitly lists tailoring to prior knowledge as unsolved.
   - Kosmos is sensitive to objective phrasing and allows no mid-run steering.
   - PDR-Bench shows agents cannot turn implicit context into a user model.
   - CellAgent cannot optimize to user preferences without manual specification.
   - No scientific agent benchmark scores personalization except PDR-Bench, which is general-domain and Chinese-annotated.
3. **Resource-aware science agents.**
   - BixBench3 shows performance collapses on datasets over 100 GB, with costs up to $525 per task.
   - AstaBench shows cost and performance trade-offs dominate.
   - Yet agents rarely reason about *which compute to use*. Incisor and the hierarchical server handle only single-node or simulated settings and list DAG workflows, user feedback and utility functions as future work.
4. **Integration across heterogeneous scholarly resources.**
   - The co-scientist lacks datasets, images and databases. AI Scientist v1 and STORM have weak citation finding. OpenScholar under-retrieves representative papers. ResearchAgent's entity store is shallow.
   - No system jointly reasons over papers, people, executable artifacts and infrastructure (see section 3).
5. **Cross-domain generalization.**
   - BioDiscoveryAgent and Biomni degrade with literature coverage. DiscoveryBench skews to social science. ScholarQABench has no social science. BioPlanner and the co-scientist call for other disciplines.
   - Most systems are single-domain, which is SciWeaver's own limitation too.
6. **Long-horizon coherence and human-in-the-loop control.**
   - AstaBench finds coding, data analysis and discovery unsolved. Kosmos needs a world model to stay coherent. Agent Laboratory and the AI Scientist hallucinate results.
   - Mixed-initiative designs (co-pilot mode, expert-in-the-loop, structured feedback UIs) are repeatedly listed as future work.
7. **Reproducibility and executable knowledge.**
   - BioCoder and ChemCrow note closed-API non-reproducibility. PaperBench and MLE-bench report contamination.
   - Paper2Agent and ToolUniverse turn papers and tools into callable MCP units, but fail on poorly documented code and do not certify correctness.
8. **Negative results and non-text evidence.** The co-scientist has no negative-results data. Weak figure and multimodal reasoning is noted by the co-scientist, BioDiscoveryAgent and PersonaAgent.

## 2. Public benchmarks SciWeaver could realistically use

| Benchmark | Fits which SciWeaver component | Size | Metric | URL |
|---|---|---|---|---|
| **AstaBench** | Whole system; literature, data and code subsets; cost-aware | 2,400+ problems | Per-benchmark scores plus cost per problem, Pareto leaderboard | arxiv.org/abs/2510.21652 |
| **LitQA2** (in LAB-Bench and AstaBench) | Knowledge Discovery agent | 248 multiple-choice questions | Precision, accuracy, source-DOI recall | arxiv.org/abs/2409.13740 |
| **ScholarQABench** | Literature synthesis (includes biomedicine and neuroscience) | 2,967 queries, 208 long-form answers | Correctness, citation precision and recall, rubric scores | arxiv.org/abs/2411.14199 |
| **LAB-Bench** (LitQA2, DbQA, ProtocolQA, SeqQA) | Literature plus the planned database-querying agent | 2,400+ multiple-choice questions (public subset) | Accuracy, precision | huggingface.co/datasets/futurehouse/lab-bench |
| **BixBench** | Notebook agent, bioinformatics analysis | 53–61 capsules, about 205–300 questions | Open-answer and multiple-choice accuracy | arxiv.org/abs/2503.00096 |
| **BixBench3** | End-to-end analysis plus compute sizing | 20 tasks, 138 artifacts | Programmatic artifact score, hours, tokens, $ | arxiv.org/abs/2608.25286 |
| **ScienceAgentBench** | Notebook and code workflows across 4 disciplines including bioinformatics | 102 tasks | Success rate, CodeBERTScore, valid execution, cost | arxiv.org/abs/2410.05080 |
| **DiscoveryBench** | Analytical intent queries | 264 real plus 903 synthetic tasks | Hypothesis-match score | github.com/allenai/discoverybench |
| **BioPlanner / BioProt** | Procedural intent (protocols) | Protocol dataset (size not recorded here) | Pseudocode precision and recall, next-step accuracy | github.com/bioplanner/bioplanner |
| **BioCoder** | Code generation within notebooks | 2,522 items | Pass@k with fuzz tests | github.com/gersteinlab/biocoder |
| **SciCode** | Cross-domain coding, including materials and physics | 80 problems, 338 subproblems | Pass@1 | scicode-bench.github.io |
| **DeepResearch Bench** | Synthesized reports and citations | 100 tasks, 22 fields | RACE (quality), FACT (citation accuracy and count) | github.com/Ayanami0730/deep_research_bench |
| **PDR-Bench** | Intent profiler and personalization | 250 user–task queries | PQR (personalization alignment, quality, factual reliability) | arxiv.org/abs/2509.25106 |
| **LaMP** | Profile-conditioned retrieval; LaMP-1 citation task is scholarly | 7 tasks | Accuracy, F1, ROUGE | arxiv.org/abs/2304.11406 |
| **SciArena / SciArena-Eval** | Validating the LLM judge against researcher votes | 20k+ votes | Judge–human pairwise agreement | arxiv.org/abs/2507.01001 |
| **Incisor's 31 HPC workloads** (protocol) | Cloud agent | 31 workloads | First-run success, cost, runtime versus expert plus SkyPilot | arxiv.org/abs/2604.24464 |

**Gaps in available benchmarks:**
- **Scholar recommendation:** no public benchmark found. Build one from co-authorship hold-out on Semantic Scholar (predict future collaborators or reviewers) [proposed, not an existing benchmark].
- **Notebook retrieval:** none found. BixBench capsules could serve as a retrieval corpus [proposal].

**Suggested evaluation package:**
1. AstaBench subsets plus LitQA2 and ScholarQABench for literature.
2. BixBench for notebooks.
3. PDR-Bench for profiling.
4. Incisor-style workloads, plus the team's materials-science runs, for cloud.
5. A judge-validation study (PaperBench JudgeEval style, or SciArena-Eval).
6. A ReAct baseline, given AstaBench's finding that plain ReAct can beat specialized workflows.

## 3. Where SciWeaver is genuinely differentiated

No system I opened combines all four resource types.

| System | Literature | People | Executable artifacts (notebooks, tools, code) | Compute recommendation with cost |
|---|---|---|---|---|
| AI co-scientist | Yes | Yes ("research contacts identification" in the Meta-review agent) | No | No |
| Biomni | Indirectly | No | Yes (150 tools, 105 packages, 59 databases) | No |
| ToolUniverse | — | No | Yes (2,700+ tools) | No |
| Kosmos, Robin | Yes | No | Yes (analysis, code execution) | No |
| Asta / AstaBench | Yes (paper finding, ScholarQA) | Not evaluated [unverified] | Yes (data, code) | Not compute selection; cost only as an evaluation metric |
| Paper2Agent | — | No | Yes (papers turned into runnable MCP tools) | No |
| Incisor, Hierarchical Server, Parsl agent | No | No | No | Yes |
| PersonaRAG, PersonaAgent, PDR-Bench | Personalization only, not scientific multi-resource | — | — | — |

**What is genuinely new:**
1. The **joint literature + people + notebooks + compute (with cost)** recommendation in one DAG-orchestrated plan.
2. **Bloom-taxonomy intent conditioning** of both decomposition and synthesis. Co-STORM, Kosmos, CellAgent and PDR-Bench all name this as an open problem.
3. A **science-gateway deployment** (OnTimeRecommend) with real users.

**What is not new:**
- Collaborator suggestion, which overlaps with the AI co-scientist.
- Multi-agent planning, judge loops and re-planning (CellAgent, AI co-scientist, Robin, AI Scientist-v2).
- Literature RAG (PaperQA2, OpenScholar).

**Biggest vulnerability for an A* submission:** the 30-query, single-domain, LLM-judge-only evaluation. The literature above (ChemCrow, BioPlanner, Agent Laboratory, PaperBench, SciArena) gives reviewers ready-made grounds to reject it. Pair the multi-resource and personalization claims with the public benchmarks in section 2. Add a human or judge-calibration study and a materials-science cross-domain split. A Cloud-agent comparison against Incisor-style baselines (first-run success, cost, runtime) would sharpen the compute novelty claim.

**Caveat on coverage:** my 2025–2026 search was targeted, not exhaustive. A system combining all four resources could exist in venues I did not search, such as SC, eScience or PEARC gateway workshops. Run a final check there before claiming "first".
