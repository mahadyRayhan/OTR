# Literature review: agentic reasoning loops and multi-agent orchestration, as related prior work for SciWeaver

**How this was checked.** I downloaded the arXiv PDF and abstract page for 40 papers and extracted the text locally. The venue, numbers, limitations and future work below come from that text unless they are marked **[unverified]**. Two places use a marker of their own:
- **[venue unverified]** means neither the PDF nor the arXiv "Comments" field states the venue. The venue shown is from memory. I could not check it against dblp or Semantic Scholar because both rate-limited me (HTTP 429).
- "No dedicated limitations section" means the arXiv text has none. A camera-ready version may have one.

---

## A. Single-agent reasoning and acting loops

**1. ReAct: Synergizing Reasoning and Acting in Language Models.** Yao et al., ICLR 2023. https://arxiv.org/abs/2210.03629
- **Core idea and result.** The model interleaves free-text reasoning traces with tool or environment actions. On ALFWorld and WebShop it beats imitation-learning and RL methods by +34% and +10% absolute success rate, using only one or two in-context examples. On HotpotQA and FEVER, querying a Wikipedia API reduces hallucination and error propagation.
- **Limitations and future work.** Complex tasks with large action spaces need more demonstrations than fit in the in-context window. Fine-tuning on HotpotQA was only a first step and needs more high-quality human annotations. They suggest scaling with multi-task training and combining with RL. Human feedback is left for future work.
- **Relevance to SciWeaver.** This is SciWeaver's main baseline, but it is the weakest kind of comparison an A* reviewer will accept. ReAct has no DAG, no typed specialists and no judge. SciWeaver's +0.46 Global Score over ReAct will read as expected rather than strong.

**2. Reflexion: Language Agents with Verbal Reinforcement Learning.** Shinn et al., NeurIPS 2023 [venue unverified]. https://arxiv.org/abs/2303.11366
- **Core idea and result.** The agent writes verbal reflections on task feedback into an episodic memory and retries. Results: 91% pass@1 on HumanEval (GPT-4 baseline 80%); 130/134 ALFWorld tasks; +22% on ALFWorld, +20% on HotpotQA and +11% on HumanEval over baselines.
- **Limitations and future work.** Verbal policy optimisation can get stuck in non-optimal local minima. Memory is a capped sliding window (last three reflections); they suggest vector or SQL memory stores. Test-driven code feedback breaks on non-deterministic, impure, hardware-dependent or concurrent functions. Future work: value learning in natural language and off-policy exploration.
- **Relevance to SciWeaver.** PoT's "enrich context and retry" is a per-node form of Reflexion. Unlike Reflexion, PoT has no cross-episode memory, so lessons are not reused across queries.

**3. Self-Refine: Iterative Refinement with Self-Feedback.** Madaan et al., NeurIPS 2023 [venue unverified]. https://arxiv.org/abs/2303.17651
- **Core idea and result.** A single LLM generates an output, critiques it and refines it in a loop. Across 7 tasks it improves task performance by about 20% absolute on average over one-step generation, for both GPT-3.5 and GPT-4.
- **Limitations (Sec. 6).** The base model needs strong few-shot or instruction-following ability. All experiments use closed models (GPT-3.5, ChatGPT, GPT-4, Codex), which hurts reproducibility and costs money. Experiments are English-only. The method has no safeguard against steering a model toward toxic output.
- **Relevance to SciWeaver.** Self-Refine is the single-agent "judge then revise" ancestor of PoT. A reviewer will ask whether PoT does better than per-agent Self-Refine with no orchestration-level re-planning. That ablation is missing.

**4. Tree of Thoughts.** Yao et al., NeurIPS 2023. https://arxiv.org/abs/2305.10601
- **Core idea and result.** The model searches with BFS or DFS over "thoughts", using its own state evaluation and backtracking. On Game of 24, GPT-4 with chain-of-thought solves 4%; ToT solves 74%.
- **Limitations and future work.** Deliberate search may be unnecessary for tasks GPT-4 already does well, and only three relatively simple tasks were studied. The method costs more than sampling. Future work: fine-tuning models with ToT-style counterfactual decision making.
- **Relevance to SciWeaver.** ToT searches within one agent's reasoning, whereas SciWeaver searches over the agent DAG. ToT's self-evaluation of thoughts is conceptually close to PoT's per-node scoring.

**5. Graph of Thoughts.** Besta et al., AAAI 2024 (arXiv journal-ref). https://arxiv.org/abs/2308.09687
- **Core idea and result.** Thoughts are vertices in an arbitrary graph and dependencies are edges, which allows operations such as aggregation and feedback loops. Sorting quality is 62% better than ToT at more than 31% lower cost. They also propose a "volume of a thought" metric.
- **Limitations.** No dedicated limitations section. The conclusion frames this as a step toward more principled prompt engineering.
- **Relevance to SciWeaver.** GoT's "aggregate and refine" graph operations are the reasoning-level counterpart of SciWeaver rewiring its DAG. GoT's graph is fixed by the user, while SciWeaver's is generated by an LLM and re-planned.

**6. Language Agent Tree Search (LATS).** Zhou et al., ICML 2024 [venue unverified]. https://arxiv.org/abs/2310.04406
- **Core idea and result.** Monte Carlo tree search (MCTS) over ReAct trajectories, with an LM value function and self-reflection. Results: 92.7% pass@1 on HumanEval (GPT-4) and an average score of 75.9 on WebShop (GPT-3.5).
- **Limitations and future work.** Compute cost is higher than ReAct or Reflexion. The method assumes the environment can be reverted to earlier states, which is not always possible. The authors expect inference costs to fall.
- **Relevance to SciWeaver.** Like PoT, LATS uses an LLM value or score to choose among alternatives. Unlike PoT, it explores several alternatives instead of repairing one node.

## B. Planner/executor separation and DAG execution

**7. ReWOO: Decoupling Reasoning from Observations.** Xu et al., arXiv 2023 (the arXiv version is marked "Preprint, under review"). https://arxiv.org/abs/2305.18323
- **Core idea and result.** A Planner writes the full tool plan up front, Workers execute it and a Solver combines the results. On HotpotQA it is 5× more token-efficient with +4% accuracy. It is robust to tool failure, and the planner can be distilled from GPT-3.5 (175B) into a 7B LLaMA.
- **Limitations and future work.** Planning everything up front fails when little is known about the environment, for example ALFWorld search. They explicitly propose wiring LLMs, tools and sub-models into a DAG. Future work: offloading specialised abilities to small models, tool representation learning, and DAG and concurrency optimisation.
- **Relevance to SciWeaver.** This is an early statement of SciWeaver's design (a planned DAG of specialists). ReWOO has no runtime verification or re-planning; PoT adds both.

**8. An LLM Compiler for Parallel Function Calling (LLMCompiler).** Kim et al., ICML 2024. https://arxiv.org/abs/2312.04511
- **Core idea and result.** A Planner emits a task DAG with dependencies, a Task-Fetching Unit dispatches ready tasks and Executors run them in parallel. Up to 3.7× lower latency, 6.7× lower cost and about 9% higher accuracy. It also re-plans dynamically.
- **Limitations.** No dedicated limitations section. The conclusion only points to future work on large-scale tasks.
- **Relevance to SciWeaver.** This is the closest prior work on typed DAG execution. Reviewers will expect SciWeaver to report latency and parallelism the way LLMCompiler does. SciWeaver adds context routing between nodes (CUGCR) and judge-triggered rewiring.

**9. HuggingGPT.** Shen et al., NeurIPS 2023 (from the PDF footer). https://arxiv.org/abs/2303.17580
- **Core idea and result.** ChatGPT acts as a controller: it plans tasks, selects Hugging Face models by their descriptions, executes them and summarises. Evaluation is mostly qualitative plus planning-quality studies.
- **Limitations (stated).**
  1. Planning quality depends on the LLM, so plans are not guaranteed feasible or optimal.
  2. Many LLM calls make it slow.
  3. The token limit restricts how many model descriptions can be included, so description summarisation needs work.
  4. LLM outputs are unstable and can cause exceptions in the program workflow.
- **Relevance to SciWeaver.** This is the original "LLM controller with a typed dependency plan over expert tools" design. Its limitations 1, 3 and 4 are exactly what PoT and CUGCR address. Cite it as the starting point.

**10. TDAG: Dynamic Task Decomposition and Agent Generation.** Wang et al., *Neural Networks* (journal, per arXiv comment). https://arxiv.org/abs/2402.10178
- **Core idea and result.** The framework decomposes tasks dynamically and generates a sub-agent for each subtask, so later subtasks adapt to earlier results. It also introduces ItineraryBench (travel planning with fine-grained, partial-credit scoring), where it beats baselines.
- **Limitations.** No dedicated limitations section in the arXiv text.
- **Relevance to SciWeaver.** TDAG adapts the decomposition as it runs. It also offers partial-credit evaluation, which SciWeaver's 0–1 judge could borrow.

**11. Agent-Oriented Planning in Multi-Agent Systems (AOP).** Li et al., ICLR 2025. https://arxiv.org/abs/2410.02189
- **Core idea and result.** A meta-agent decomposes the query and assigns subtasks to agents, guided by three principles: solvability, completeness and non-redundancy. A reward model scores each sub-task's response. The meta-agent then re-plans, plans in more detail or re-describes the sub-task, with a feedback loop. On Husky-QA it beats GPT-4o, CoT and zero-shot CoT by 10.4%, 8.1% and 11.5% accuracy, and beats other meta-agent multi-agent systems by at least 4.1% at similar cost. A dependency detector adds +2.5% (43.7 → 46.2).
- **Limitations (App. F.4).** The number of agents is bounded by the LLM context window. They propose grouping agents hierarchically. Cost is much higher than single-agent systems.
- **Relevance to SciWeaver.** **This is the most direct prior art for PoT.** The structure matches: decompose, assign to agents, score each sub-task response with an evaluator, then re-plan or re-describe. SciWeaver's novelty must rest on its typed failure classes (specificity, coverage, relevance, knowledge-boundary) mapped to specific repairs, on CUGCR, and on user-intent conditioning. It needs an explicit comparison with AOP.

**12. Flow: Modularized Agentic Workflow Automation.** Niu et al., ICLR 2025 (the repo cited in the paper is "2025_ICLR_FLOW"). https://arxiv.org/abs/2501.07834
- **Core idea and result.** The workflow is an activity-on-vertex (AOV) graph that LLM agents refine during execution. Subtask allocation is adjusted based on how subtasks have performed so far, and candidate workflows are chosen for modularity (parallelism and dependency complexity). It is evaluated on website design, Gobang game development and LaTeX Beamer writing, with human rankings, and shows better efficiency and error tolerance.
- **Limitations (App. F).** Choosing the most modular of several candidate workflows is still not the most efficient approach; they suggest training a workflow-management model with KPI rewards. The updater needs global information, which becomes a problem as context grows; they suggest hierarchical or targeted updates.
- **Relevance to SciWeaver.** Flow is runtime DAG updating, very close to PoT's rewiring. Its global-context limitation is the problem CUGCR is meant to solve.

## C. Hand-designed multi-agent frameworks

**13. CAMEL: Communicative Agents for "Mind" Exploration.** Li et al., NeurIPS 2023. https://arxiv.org/abs/2303.17760
- **Core idea and result.** Two role-playing agents (user and assistant) cooperate under "inception prompting". The work mainly contributes data and analysis. It identifies failure modes: conversation deviation, role flipping and termination problems.
- **Limitations (App. K.1).** Evaluating many diverse generated tasks would need many domain experts. Human and LLM evaluators may be biased (humans prefer longer answers). Only two agents were studied; future work covers more agents and competitive settings.
- **Relevance to SciWeaver.** CAMEL's warning about evaluator bias applies directly to SciWeaver's LLM-as-judge results.

**14. AutoGen: Next-Gen LLM Applications via Multi-Agent Conversation.** Wu et al., COLM 2024 [venue unverified]. https://arxiv.org/abs/2308.08155
- **Core idea and result.** A framework of conversable agents with "conversation programming". On MATH with GPT-4 it scores 69.48% versus 55.18% for vanilla GPT-4, plus other case studies.
- **Future work (stated).**
  - Design optimal multi-agent workflows: how many agents, which roles and how they interact.
  - Build highly capable agents (CAMEL fails without code or tool execution).
  - Find the right balance between automation and human control.
  - Tools to log and debug complex workflows, to avoid "unintelligible chatter".
- **Relevance to SciWeaver.** These future-work items are what SciWeaver's Orchestrator and PoT partly answer. AutoGen GroupChat is also a standard baseline.

**15. MetaGPT.** Hong et al., ICLR 2024. https://arxiv.org/abs/2308.00352
- **Core idea and result.** Standard operating procedures (SOPs) are encoded as role pipelines (PM → architect → engineer) with structured artefacts and executable feedback. Pass@1 is 85.9% on HumanEval and 87.7% on MBPP.
- **Limitations.** No UI or front-end agents and no multimodal tools. Hard to meet diverse real-world requirements. Users cannot interrupt an agent or set checkpoints.
- **Relevance to SciWeaver.** MetaGPT's typed artefacts between roles are a hand-written alternative to CUGCR's "information atoms".

**16. AgentVerse.** Chen et al., ICLR 2024 [venue unverified; arXiv says "under review"]. https://arxiv.org/abs/2308.10848
- **Core idea and result.** Four stages: expert recruitment, collaborative decision-making, action execution and evaluation. The evaluation loop adjusts group composition. Beats single agents on reasoning, coding, tool use and Minecraft, with emergent positive and harmful behaviours.
- **Limitations and future work.** Needs more capable or multimodal agents. Multi-party communication is hard because agents must decide when to speak and to whom. Positive emergent behaviours should be exploited and harmful ones mitigated.
- **Relevance to SciWeaver.** AgentVerse's evaluator-driven re-recruitment resembles PoT re-planning, but works at the level of the whole group rather than per node.

**17. Magentic-One.** Fourney et al., Microsoft tech report 2024 (arXiv only). https://arxiv.org/abs/2411.04468
- **Core idea and result.** An Orchestrator keeps task and progress "ledgers", plans, tracks progress and re-plans to recover from errors, directing WebSurfer, FileSurfer, Coder and Terminal agents. With GPT-4o plus o1: 38.0% on GAIA test, 27.7% on AssistantBench and 32.8% on WebArena, statistically competitive with the state of the art.
- **Limitations (Sec. 6.2).**
  - Evaluation measures accuracy only, ignoring cost, latency and user preference.
  - High cost and latency: dozens of calls, several dollars and tens of minutes per task.
  - Limited modalities and action space.
  - A weak Coder; they suggest a notebook-style design.
  - Fixed team membership: unneeded agents distract the orchestrator.
  - No learning across tasks: sub-task solutions are rediscovered repeatedly. Long-term memory is the key future direction.
- **Relevance to SciWeaver.** **This is a must-compare baseline**: an orchestrator with re-planning over specialists. Several of SciWeaver's design choices respond to its stated limitations: per-query team selection through the DAG, the user-preference profiler, and token-budgeted routing.

**18. Scaling LLM-based Multi-Agent Collaboration (MacNet).** Qian et al., ICLR 2025. https://arxiv.org/abs/2406.07155
- **Core idea and result.** Agents are organised in DAGs (chain, tree, mesh, irregular) and collaborate in topological order. It scales to more than 1,000 agents. Irregular topologies beat regular ones. They find a "collaborative scaling law": performance grows logistically with the number of agents.
- **Limitations.** No dedicated section. The conclusion notes there are limits to the scaling horizon.
- **Relevance to SciWeaver.** MacNet provides evidence that DAG shape matters, which justifies an LLM-generated typed DAG rather than a fixed chain.

## D. Learned topologies and automated workflow search (offline, per task)

**19. DyLAN: Dynamic LLM-Powered Agent Network.** Liu et al., COLM 2024. https://arxiv.org/abs/2310.02170
- **Core idea and result.** Two stages. Team Optimization selects agents with an unsupervised Agent Importance Score from a preliminary trial. Task Solving then runs a dynamic feed-forward network that can stop early. Team selection improves accuracy by up to 25.0% on some MMLU subjects.
- **Future work.** Testing on open-source foundation models.
- **Relevance to SciWeaver.** The Agent Importance Score is a credit-assignment signal that PoT's per-node scores could be calibrated against.

**20. GPTSwarm: Language Agents as Optimizable Graphs.** Zhuge et al., ICML 2024. https://arxiv.org/abs/2402.16823
- **Core idea and result.** Agents are computational graphs. Node prompts are optimised and edges are optimised with REINFORCE. Gains on MMLU, Mini Crosswords, HumanEval and GAIA.
- **Limitations (App. G).** Only edges between agents are optimised, not the topology inside each agent. "Dynamically changing the topology may enhance task planning." Communication efficiency and robustness break down beyond about 100 agents.
- **Relevance to SciWeaver.** GPTSwarm's own future work is dynamic topology change, which is what PoT's rewiring does at runtime and without training.

**21. G-Designer.** Zhang et al., ICML 2025 [venue unverified]. https://arxiv.org/abs/2410.11782
- **Core idea and result.** A variational graph autoencoder designs a communication topology adapted to each task. Results: 84.50% on MMLU and 89.90% pass@1 on HumanEval, up to 95.33% fewer tokens, and only a 0.3% drop under adversarial agents.
- **Limitations.** No dedicated section found.
- **Relevance to SciWeaver.** G-Designer generates a query-conditioned topology with a learned graph network, which is a learned alternative to SciWeaver's LLM decomposer.

**22. AgentPrune ("Cut the Crap").** Zhang et al., 2024 (ICLR 2025 [venue unverified]). https://arxiv.org/abs/2410.02506
- **Core idea and result.** One-shot pruning of the spatial-temporal message-passing graph to remove redundant or malicious messages. It matches state-of-the-art topologies at $5.6 versus $43.7, cuts tokens by 28.1–72.8%, and defends against two agent attacks.
- **Limitations.** None stated in the arXiv text.
- **Relevance to SciWeaver.** **This is the key baseline for CUGCR**: both reduce the tokens passed between agents. CUGCR selects content at the level of information atoms under a budget, while AgentPrune removes whole messages and edges.

**23. Automated Design of Agentic Systems (ADAS / Meta Agent Search).** Hu, Lu and Clune, ICLR 2025. https://arxiv.org/abs/2408.08435
- **Core idea and result.** A meta-agent writes new agents in code, drawing on a growing archive of earlier designs. Gains: +13.6 F1 on DROP and +14.4% on MGSM. After transfer across domains: +25.9% on GSM8K and +13.2% on GSM-Hard.
- **Future work (Sec. 6).**
  - Higher-order ADAS, in which the meta-agent itself is optimised.
  - Online continual learning from deployment feedback.
  - Multi-objective ADAS (cost, latency, robustness).
  - Safety: sandboxed execution of generated code.
- **Relevance to SciWeaver.** ADAS designs the system offline for a task, while SciWeaver adapts each query at runtime. ADAS's "online continual learning" future work is open ground for SciWeaver's user profiler.

**24. AFlow: Automating Agentic Workflow Generation.** Zhang et al., ICLR 2025. https://arxiv.org/abs/2410.10762
- **Core idea and result.** MCTS over workflows written as code, built from operators, with execution feedback. +5.7% average over the state of the art on six benchmarks. Smaller models beat GPT-4o at 4.55% of the cost.
- **Limitations.** No dedicated section. App. F discusses applications.
- **Relevance to SciWeaver.** AFlow is the standard automated-workflow baseline. Its workflows are fixed per task, while SciWeaver's are per query and repaired at runtime.

**25. MASS: Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies.** Zhou et al. (Google), ICLR 2026 (arXiv comment). https://arxiv.org/abs/2502.02533
- **Core idea and result.** Three optimisation stages: block-level prompts, then workflow topology in a pruned space, then workflow-level prompts. Main finding: prompts matter more than topology. It beats manual and automated multi-agent systems on MATH, DROP, HotpotQA, MuSiQue, 2Wiki, MBPP, HumanEval and LiveCodeBench.
- **Limitations (App. A).** The topology space could be extended, for example with sparse debate. More sample-efficient search (Bayesian optimisation) is needed. The prompt optimiser could use textual feedback from error logs.
- **Relevance to SciWeaver.** A reviewer may ask whether PoT's gains come from better prompts, meaning context enrichment, rather than from rewiring. SciWeaver needs that ablation.

**26. EvoAgent.** Yuan et al., NAACL 2025. https://arxiv.org/abs/2406.14228
- **Core idea and result.** Evolutionary operators (mutation, crossover, selection) expand a single specialised agent into a multi-agent system. Gains on reasoning, MMMU and TravelPlanner.
- **Limitations.** Higher token cost than a single agent. Limited comparison with other multi-agent frameworks. Generated agents were not evaluated manually.
- **Relevance to SciWeaver.** Moderate. EvoAgent is automatic team generation; SciWeaver's team of four specialists is fixed.

**27. EvoMAC: Self-Evolving Multi-Agent Collaboration Networks for Software Development.** Hu et al., ICLR 2025 [venue unverified]. https://arxiv.org/abs/2410.16946
- **Core idea and result.** Test-time "textual backpropagation" from verifier feedback updates agents and their connections. It introduces rSDE-Bench. Beats the state of the art by 26.48% on Website Basic, 34.78% on Game Basic and 6.10% on HumanEval.
- **Future work.** A reward model to improve learning from feedback; more software types.
- **Relevance to SciWeaver.** EvoMAC is test-time rewiring from verifier feedback, conceptually very close to PoT, but in a domain (code) where an executable verifier exists. SciWeaver's open-ended science queries have no such verifier, which is both its challenge and a possible novelty.

**28. ScoreFlow.** Wang et al., 2025 [venue unverified]. https://arxiv.org/abs/2502.04306
- **Core idea and result.** A workflow generator is trained with Score-DPO, which uses quantitative evaluation scores in preference optimisation. +8.2% over baselines on six QA, code and math benchmarks.
- **Limitations.** No dedicated section found.
- **Relevance to SciWeaver.** ScoreFlow shows how judge scores could train the Query Decomposer instead of being used only at runtime.

## E. Per-query system generation, routing and learned orchestrators

**29. MaAS: Multi-agent Architecture Search via Agentic Supernet.** Zhang et al., 2025 (ICML 2025 [venue unverified]). https://arxiv.org/abs/2502.04180
- **Core idea and result.** It learns a probabilistic "agentic supernet" and samples a system for each query, with early exit for easy queries. Inference cost is 6–45% of existing systems, performance is +0.54–11.82% (abstract; the PDF text says 0.54–16.89%), and it transfers across datasets and backbones.
- **Limitations.** No dedicated section.
- **Relevance to SciWeaver.** **A strong competitor on query adaptation.** MaAS sets the depth of each query's system by difficulty. SciWeaver adapts by user intent and dependency type, which MaAS does not model.

**30. MasRouter.** Yue et al., 2025 (ACL 2025 [unverified; only a secondary listing found]). https://arxiv.org/abs/2502.11133
- **Core idea and result.** A cascaded controller decides the collaboration mode, assigns roles and routes each role to an LLM. +1.8–8.2% on MBPP, up to 52.07% lower cost on HumanEval, and 17.21–28.17% savings when plugged into existing frameworks.
- **Limitations.** None found.
- **Relevance to SciWeaver.** SciWeaver routes to fixed agents and models. Choosing the LLM per node is a natural extension.

**31. MAS-GPT.** Ye et al., 2025 (ICML 2025 [venue unverified]). https://arxiv.org/abs/2503.03686
- **Core idea and result.** An LLM is fine-tuned to output an executable multi-agent system (as code) for each query in a single inference. Across 9 benchmarks and 5 LLMs it beats more than 10 multi-agent baselines, by +3.89% over the second-best on average.
- **Limitations.** No dedicated section.
- **Relevance to SciWeaver.** Generating a whole system per query in one call is a cheap alternative to SciWeaver's decomposer plus re-planner.

**32. FlowReasoner: Reinforcing Query-Level Meta-Agents.** Gao et al., 2025 [venue unverified]. https://arxiv.org/abs/2504.15257
- **Core idea and result.** A meta-agent distilled from DeepSeek-R1 and trained with RL on execution feedback, rewarding performance, complexity and efficiency. It designs one system per query and is +10.52% over o1-mini across three code benchmarks.
- **Limitations (observed in the paper).** Open-source models produce unreliable workflows from instructions alone and depend on hand-made workflows. All benchmarks are code.
- **Relevance to SciWeaver.** FlowReasoner's reward combines performance, complexity and efficiency, which maps onto a judge score plus a token budget. It could train SciWeaver's decomposer.

**33. MAS-Zero.** Ke et al., SEA@NeurIPS 2025 workshop (Oral), per the arXiv comment. https://arxiv.org/abs/2505.14996
- **Core idea and result.** At inference time a meta-level loop designs, critiques and refines the multi-agent system for each problem, with no validation set. Meta-feedback covers solvability and completeness, and the system can fall back to simpler designs. Gains of up to +16.69% on reasoning, +16.66% on coding and +5.45% on agentic tasks.
- **Limitations.** No dedicated section found.
- **Relevance to SciWeaver.** **Very close to PoT**: inference-time critique and redesign. Its solvability and completeness feedback overlaps with PoT's coverage and knowledge-boundary failure types. Its "reduce to a simpler system" option is something SciWeaver lacks.

**34. Puppeteer: Multi-Agent Collaboration via Evolving Orchestration.** Dang et al., NeurIPS 2025. https://arxiv.org/abs/2505.19591
- **Core idea and result.** A central orchestrator trained with RL picks the next agent at each step, rewarded for accuracy minus a cost penalty. Compact, cyclic reasoning structures emerge. Better quality at lower cost on GSM-Hard, MMLU-Pro, SRDD and CommonGen-Hard.
- **Limitations (stated).** Rewards are coarse (final output and tokens only) and need step-level supervision. The set of agents and tools is fixed, and dynamic modification at inference is needed. There is occasional mis-coordination or deceptive agreement between agents.
- **Relevance to SciWeaver.** PoT supplies exactly the step-level signal Puppeteer says it lacks. This is a strong positioning argument for SciWeaver.

**35. COLLATOR (earlier titled LEMON): Compositional Multi-Agent Orchestration with Counterfactual RL.** Chen et al., arXiv 2026, submitted to ICLR 2027. https://arxiv.org/abs/2605.14483
- **Core idea and result.** A GRPO-trained orchestrator writes a single specification of duties, capacity levels and dependencies. A counterfactual credit signal edits one field at a time and applies the reward difference only to the edited spans. Best average across six reasoning and code benchmarks, with a better accuracy-token trade-off.
- **Limitations.** Training is expensive because it executes workers repeatedly (counterfactuals add 21.6% training tokens). Evaluation covers text reasoning and code only; long-horizon tool use, multimodal and interactive settings are future work.
- **Relevance to SciWeaver.** Its node-local counterfactual credit is the trained counterpart of PoT's per-node diagnosis.

## F. Failure analysis, attribution and repair (most relevant to PoT)

**36. Why Do Multi-Agent LLM Systems Fail? (MAST).** Cemri et al., NeurIPS 2025 Datasets & Benchmarks track (PDF footer). https://arxiv.org/abs/2503.13657
- **Core idea and result.** More than 1,600 annotated traces from 7 frameworks. The taxonomy has 14 failure modes in 3 categories: system design, inter-agent misalignment and task verification. Annotator agreement is κ=0.88, and an LLM-as-judge annotator is released. In 210 traces: system design 41.8%, misalignment 36.9%, verification 21.3%. Example modes: step repetition 15.7%, ignoring task requirements 11.8%, not recognising task completion 12.4%. Adding high-level objective verification to ChatDev gives +15.6%.
- **Limitations and future work.** Simple interventions do not fix every failure mode and completion rates stay low. They argue "combinatorial changes" are needed, from agent organisation to model improvements.
- **Relevance to SciWeaver.** **Essential.** PoT's four failure types should be mapped onto MAST, and SciWeaver should be evaluated with MAST-Data or the MAST annotator. Right now PoT's taxonomy looks ad hoc.

**37. Which Agent Causes Task Failures and When? (Who&When).** Zhang et al., 2025 (arXiv "camera-ready"; ICML 2025 [venue unverified]). https://arxiv.org/abs/2505.00212
- **Core idea and result.** Defines automated failure attribution, meaning identifying the responsible agent and step. Dataset: 127 multi-agent systems. The best method reaches 53.5% agent-level accuracy but only 14.2% step-level. o1 and R1 are not practically usable.
- **Limitations.** No dedicated section. The paper stresses that the task is very hard.
- **Relevance to SciWeaver.** PoT assumes per-node judging finds the failing node. Who&When suggests this is unreliable, so PoT's localisation accuracy must be measured.

**38. Where LLM Agents Fail and How They Can Learn From Failures (AgentDebug).** Zhu et al., arXiv Sept 2025 (an ICML 2026 listing appeared in search [unverified]). https://arxiv.org/abs/2509.25370
- **Core idea and result.** AgentErrorTaxonomy spans memory, reflection, planning, action and system failures. AgentErrorBench is built from ALFWorld, GAIA and WebShop. AgentDebug finds the root-cause error and gives targeted feedback: +24% all-correct and +17% step accuracy over the best baseline, and up to +26% relative task success.
- **Limitations.** The benchmark is limited in scale and domain (multimodal, long-horizon and safety-critical domains such as healthcare are future work). Training a dedicated debugger is too costly for academic labs, so prompting may underperform a trained model.
- **Relevance to SciWeaver.** "Root cause, then targeted feedback" is the same pattern as PoT's diagnosis followed by enrichment. AgentDebug calls healthcare-like domains open, which supports a biomedical setting for SciWeaver.

**39. DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems.** Ma et al. (Microsoft), arXiv Dec 2025 / Jan 2026. https://arxiv.org/abs/2512.06749
- **Core idea and result.** Debugging hypotheses are tested by actually intervening (editing messages or plans) and replaying. It reframes attribution as an ill-posed problem, since several different interventions can each repair a failure. On Magentic-One with GAIA and AssistantBench it turns 18–28% of failures into successes. On AG2 with GSMPlus it recovers 49%.
- **Limitations.** Shown only as feasible on two frameworks and on sequential or centrally orchestrated topologies. Requires rich logs and a checkpoint/replay interface. Not tested on production, latency-constrained or safety-critical settings.
- **Relevance to SciWeaver.** PoT's "enrich vs. rewire" choice is a form of intervention. DoVer argues success should be measured by whether the fix works, not by attribution accuracy, which is a good metric for PoT.

**40. VeriMAP: Verification-Aware Planning for Multi-Agent Systems.** Xu et al. (Purdue / Megagon), arXiv Oct 2025; listed at EACL 2026 (aclanthology.org/2026.eacl-long.353, per search result; the arXiv comment says ARR submission). https://arxiv.org/abs/2510.17109
- **Core idea and result.** The planner builds a subtask DAG and writes verification functions for each subtask, in Python or natural language. Failed nodes are retried; after all retries fail, it re-plans using the execution history. The planner is GPT-4.1 and executors are GPT-4o-mini. It beats single- and multi-agent baselines, e.g. 93.92% and 93.60% on two datasets (Table 1). Removing re-planning (-1it variants) consistently hurts.
- **Limitations.** Verification can reinforce flawed reasoning if the functions are poorly specified. Biases are inherited from LLMs. The planning-plus-verification chain reduces interpretability for users. Compute and energy costs are high. Re-planning simply appends the full history to the prompt; future work should "analyze specific failure signals, such as error types or patterns across retries" for targeted corrections.
- **Relevance to SciWeaver.** **A closest competitor.** VeriMAP's stated future work is essentially PoT's typed failure classification driving targeted repair. SciWeaver should cite this as the gap it fills and compare head-to-head.

**41. VMAO: Verified Multi-Agent Orchestration (Plan-Execute-Verify-Replan).** Zhang et al., ICLR 2026 Workshop on MALGAI. https://arxiv.org/abs/2603.11445
- **Core idea and result.** The query is decomposed into a DAG of sub-questions. Domain agents run in parallel with automatic context propagation. An LLM verifier checks whether the combined result is complete, and the system re-plans with configurable stop conditions. On 25 expert-curated market-research queries, judged on a 1–5 scale, completeness goes from 3.1 to 4.2 and source quality from 2.6 to 4.1 versus a single agent. The biggest gain (+53% completeness) is on open-ended queries. Most re-planning actions are retries rather than new decompositions.
- **Limitations.** The verifier checks completeness, not truth. Poor decomposition propagates errors. It costs 8.5× the tokens of a single agent (850K vs 100K). Only the Claude family was tested. Contributions of individual components were not isolated.
- **Relevance to SciWeaver.** **The nearest twin.** It has the same pipeline shape and the same weak evaluation design (a small expert query set judged by an LLM; SciWeaver has 30 queries). A reviewer will see SciWeaver as VMAO plus per-node typed diagnosis, CUGCR and a user profiler. SciWeaver needs a stronger evaluation than VMAO to stand out.

**42. Self-Healing Agentic Orchestrators for Reliable Tool-Augmented LLM Systems.** Babu (independent researcher), arXiv May 2026. https://arxiv.org/abs/2606.01416
- **Core idea and result.** Observed failure signals are mapped to failure classes, which select targeted recovery actions within budgets; recovered trajectories are verified. On a 100-task fault-injection benchmark: 98.8% success, versus 94.5% for retry-only and 93.8% for full re-planning. With a single recovery attempt: 94.0% versus 85.3% and 88.2%. Verifier-guided repair cuts silent failures to 0%.
- **Limitations (threats to validity).** Controlled benchmark with deterministic tools and injected faults, so it does not establish production reliability.
- **Relevance to SciWeaver.** Its "failure class → targeted action" mapping is very close to PoT. It is also evidence that targeted repair beats full re-planning, which can support PoT's enrich-vs-rewire design. It is not an A* venue paper.

**43. A Self-Healing Framework for Reliable LLM-Based Autonomous Agents.** Jeong and Shin (Samsung SDS), arXiv May 2026. https://arxiv.org/abs/2605.06737
- **Core idea and result.** A failure taxonomy, a reliability assessment model, detection from execution patterns and output consistency, and recovery by adaptive re-planning and corrective prompting. The abstract reports higher success with no numbers.
- **Limitations.** None found.
- **Relevance to SciWeaver.** A weak but directly overlapping prior. Cite it to head off novelty objections.

**44. FlowFixer: Diagnosis-Driven Automatic Repair for Agentic Workflow via Symbolic Inference.** Ma et al., arXiv July 2026 (cs.SE). https://arxiv.org/abs/2607.02882
- **Core idea and result.** Workflow executions on Dify, Coze and n8n are turned into symbolic traces. It infers specifications for node correctness, temporal and causal dependencies, attributes failures to root causes and generates repair patches. Repair success is 71.3%, +11.9–27.6% over the state of the art. Attribution improves 4.8–33.1% and root-cause analysis 15.3–38.8%.
- **Limitations.** No dedicated section found.
- **Relevance to SciWeaver.** FlowFixer works on node-level specifications, which suggests PoT's LLM judge could be backed by symbolic or specification checks.

## G. Test-time topology adaptation (2026)

**45. TacoMAS.** Xu et al., arXiv May 2026. https://arxiv.org/abs/2605.09539
- **Core idea and result.** Inference is treated as online graph adaptation. A fast loop updates agent capabilities from trajectory feedback; a slow loop, run by a meta-LLM, edits edges and adds or removes agents. They argue this converges to a stable task-conditioned equilibrium. On Finance, BrowseComp-Plus, PlanCraft and WorkBench it beats about 20 baselines by 13.3% on average.
- **Limitations (App. B).** A single meta-controller sees the full trajectory and becomes a bottleneck for long or highly decomposable tasks; they propose hierarchical controllers. Scratch memory is cleared between instances, so structural discoveries are rediscovered; cross-instance memory with privacy safeguards is future work.
- **Relevance to SciWeaver.** PoT's "enrich context" step is a fast-loop capability update and "rewire DAG" is a slow topology update. TacoMAS shows that separating these time scales matters, so SciWeaver should frame and test PoT that way.

**46. MANTA: Multi-Agent Network Topology Adaptation.** Huang et al. (Cornell et al.), arXiv July 2026. https://arxiv.org/abs/2607.28527
- **Core idea and result.** Starts from a task-conditioned topology based on past structural experience. It audits collaboration traces and makes bounded structural repairs: roles, links, execution order, information visibility and validation paths, all within a fixed agent budget. Average score 74.0, +5.8 points over the best baseline, across five benchmarks including PlanCraft.
- **Limitations.** No dedicated section found.
- **Relevance to SciWeaver.** **A strong competitor for PoT's DAG rewiring.** Its "information visibility" edits overlap with CUGCR's routing.

**47. AdaptOrch.** Yu, arXiv Feb 2026. https://arxiv.org/abs/2602.16873
- **Core idea and result.** Routes each task's decomposition DAG to one of four topologies (parallel, sequential, hierarchical, hybrid) in O(|V|+|E|) time, with a "performance convergence scaling law". +12–23% over single static topologies on SWE-bench, GPQA and RAG.
- **Limitations.** Quality depends on the decomposer. Coupling estimates are coarse. Parallel calls can hit rate or budget limits. Only three benchmarks.
- **Future work.** Learned routing, dynamic re-orchestration mid-execution, cost-aware routing, cross-modal use.
- **Relevance to SciWeaver.** AdaptOrch's "dynamic re-orchestration mid-execution" future work is what PoT already does.

## H. Inter-agent context management (relevant to CUGCR)

**48. Agent-Radar: Attention Steering with Context Relevance.** Zhang et al. (Purdue), arXiv May 2026. https://arxiv.org/abs/2605.30136
- **Core idea and result.** A training-free method that steers each agent's attention toward relevant context using temporal and spatial decay, instead of compressing or pruning. Up to +7.64 points over compression and pruning methods on five benchmarks. It stays robust as the number of agents and rounds grows.
- **Limitations.** In an appendix I did not extract **[unverified]**.
- **Relevance to SciWeaver.** CUGCR must be compared with AgentPrune (pruning) and Agent-Radar (attention steering) at matched token budgets.

## I. Bioinformatics multi-agent systems (domain competitors, outside the core theme)

**49. K-Dense Analyst.** Li et al., arXiv Aug 2025. https://arxiv.org/abs/2508.07043
- **Core idea and result.** A hierarchical multi-agent system with a dual loop (planning plus validated execution). 29.2% on BixBench, versus 22.9% for GPT-5 and 18.3% for its own backbone, Gemini 2.5 Pro.
- **Limitations.** Not extracted **[unverified]**.
- **Relevance to SciWeaver.** A bioinformatics orchestrator evaluated on a public benchmark, which SciWeaver lacks.

**50. BioAgents.** Mehandru et al., arXiv Jan 2025. https://arxiv.org/abs/2501.06314
- **Core idea and result.** A multi-agent system of fine-tuned small language models with RAG. It performs comparably to human experts on conceptual genomics tasks; its code generation needs improvement.
- **Limitations.** Code generation (from the abstract).
- **Relevance to SciWeaver.** Domain overlap. BioMaster (bioRxiv 2025; Plan, Task, Debug and Check agents) is **[unverified, seen only in search snippets]**.

---

## 1. Gaps: open problems recurring across these papers (late 2026)

1. **Step- or node-level credit assignment is unsolved.** Puppeteer has only coarse final-output rewards and asks for step-level supervision. Who&When reaches 14.2% step-level attribution. DoVer argues single-step attribution is ill-posed. COLLATOR needs expensive counterfactual edits. VeriMAP's re-planning just appends the full history. MAST finds simple interventions insufficient. PoT's per-node judge is a candidate answer, but only if its localisation accuracy is measured, for example on Who&When or MAST-Data.

2. **Verifiers and LLM judges are unreliable.** VMAO's verifier checks completeness, not truth. VeriMAP warns that badly specified verification functions reinforce flawed reasoning. MAST has a whole "task verification" failure category (21.3%). CAMEL notes biased LLM and human evaluators. Self-healing orchestrators (#42) report wrong-but-plausible "silent failures" without verifiers. ARE (arXiv 2509.17158, seen only in search **[unverified]**) reports that an RL-trained agent learned to exploit its LLM-judge reward. SciWeaver uses an LLM judge both inside the loop (PoT) and for evaluation, which is double exposure. Calibration and agreement with human experts are needed.

3. **Re-planning during execution without instability.**
   - Asked for as future work or a limitation: AdaptOrch (dynamic re-orchestration), Puppeteer and Magentic-One (fixed team), GPTSwarm (dynamic topology), Flow (updater needs global context), AOP (agent count limited by context window).
   - Started in 2026: TacoMAS (fast/slow time scales; its single meta-controller is a bottleneck) and MANTA (bounded edits).
   - Still missing: theory or criteria for when to enrich, retry, rewire or simplify (MAS-Zero's fallback to simpler systems). VMAO observed that most of its re-planning was retries rather than new decompositions.

4. **Cost and latency of orchestration.** Magentic-One (dollars and tens of minutes per task), VMAO (8.5× tokens), LATS and ToT (search cost), HuggingGPT (many calls), EvoAgent and AOP (higher cost), AdaptOrch (rate limits). ADAS lists multi-objective design as future work. MaAS, MasRouter and FlowReasoner treat cost as part of the objective. SciWeaver must report tokens, calls and latency per query, including the overhead of PoT re-planning.

5. **Context growth and inter-agent communication.** HuggingGPT's token limits on model descriptions, Flow's global-context updater, AOP's context limit on agent count, Reflexion's sliding-window memory, and TacoMAS's controller summarising "too much local information". AgentPrune and Agent-Radar address it partly. Budgeted, dependency-aware routing of content (CUGCR) is an open niche, but must be compared against these two.

6. **No learning across tasks or sessions.** Magentic-One ("discarded between tasks… key direction"), TacoMAS (scratch memory cleared), ADAS (online continual learning), Reflexion (better memory stores), MANTA (uses "cross-run experience" and is an early answer). SciWeaver's EMA preference profiler is user memory, not memory of which DAGs and repairs worked. Extending it to structural memory is a clear follow-up.

7. **Evaluation beyond accuracy, on open-ended tasks, at small scale.** Magentic-One says accuracy-only evaluation ignores cost, latency and user preference, and that open-ended tasks remain open. VMAO uses 25 queries and one model family. DoVer and the self-healing papers offer only feasibility evidence or controlled benchmarks. AdaptOrch has three benchmarks. AgentDebug notes limited domain diversity. Personalisation by user intent is essentially absent from orchestration work; the closest is Orion (AAAI 2026, single web agent **[unverified, search only]**). This is a novelty opening for SciWeaver's profiler, but it needs a proper user study.

8. **Generalisation across model families and to open models.** DyLAN (open models are future work), FlowReasoner (open models produce unreliable workflows), VMAO (Claude only), Self-Refine (closed models only), DoVer (GPT-4o and Qwen3 only).

## 2. Closest competitors to SciWeaver

1. **VMAO (arXiv 2603.11445, ICLR 2026 workshop).** Nearly the same architecture: DAG of sub-questions, domain agents, context propagation, LLM verifier, adaptive re-planning, stop conditions, and evaluation on a small expert query set with LLM judging. SciWeaver must show what per-node typed diagnosis and CUGCR add over orchestration-level verification.
2. **VeriMAP (arXiv 2510.17109, EACL 2026).** A DAG planner with verification for each subtask and re-planning. Its stated future work ("analyze specific failure signals, such as error types") is PoT's contribution, so it is the natural head-to-head baseline.
3. **AOP (ICLR 2025).** A meta-agent decomposes and assigns work, a reward model scores sub-task outputs, and it re-plans or re-describes. This is the published A* prior closest to PoT plus the Query Decomposer.
4. **Magentic-One (with DoVer on top).** The standard generalist orchestrator with re-planning over specialist agents. Reviewers will expect it as a baseline instead of, or in addition to, ReAct.
5. **MANTA / TacoMAS (2026)**, with **MaAS** as the published query-adaptive baseline. These are the current state of the art in test-time structural adaptation and per-query architecture. PoT's DAG rewiring must be positioned against them.

Also cite: AgentPrune and Agent-Radar for CUGCR, MAST for the failure taxonomy, and K-Dense Analyst (BixBench) as the bioinformatics comparison.

## 3. Standard benchmarks in this area

URLs marked ✓ appeared in a paper I opened. All others are from memory **[unverified URL]**.

| Benchmark | What it measures | URL |
|---|---|---|
| GAIA | Generalist assistant tasks: web, files, multimodal, tools | https://arxiv.org/abs/2311.12983 |
| AssistantBench | Realistic time-consuming web tasks | https://arxiv.org/abs/2407.15711 |
| WebArena | Autonomous web navigation in realistic sites | https://webarena.dev |
| HotpotQA | Multi-hop QA (used by ReAct, ReWOO, LATS, MASS) | https://hotpotqa.github.io |
| MuSiQue / 2WikiMultiHopQA | Multi-hop long-context QA (MASS, via LongBench) | https://arxiv.org/abs/2108.00573 / https://arxiv.org/abs/2011.01060 |
| ALFWorld | Text-based embodied decision-making | https://alfworld.github.io |
| WebShop | Web shopping navigation | https://webshop-pnlp.github.io |
| HumanEval / MBPP | Function-level code generation | https://github.com/openai/human-eval ; https://arxiv.org/abs/2108.07732 |
| LiveCodeBench | Contamination-free code | https://livecodebench.github.io |
| MATH / GSM8K / GSM-Hard / MGSM | Math reasoning | https://github.com/hendrycks/math ; https://github.com/openai/grade-school-math |
| DROP | Discrete reasoning over paragraphs (ADAS, MASS) | https://allenai.org/data/drop |
| MMLU / MMLU-Pro / GPQA | Knowledge and graduate-level QA | https://arxiv.org/abs/2009.03300 ; https://arxiv.org/abs/2406.01574 ; https://arxiv.org/abs/2311.12022 |
| SRDD / rSDE-Bench | Software-level development from requirements (ChatDev, Puppeteer, EvoMAC) | ✓ https://yuzhu-cai.github.io/rSDE-Bench/ |
| PlanCraft, WorkBench, BrowseComp-Plus, Finance (SEC filings) | Planning, workplace tool workflows, multi-hop retrieval, financial analysis (TacoMAS, MANTA) | ✓ named in TacoMAS; URLs unverified |
| Husky-QA | Multi-tool numerical and commonsense QA (AOP) | ✓ used in AOP |
| ItineraryBench | Fine-grained, progressively harder travel planning (TDAG) | ✓ introduced in TDAG |
| MAST-Data | 1,600+ annotated multi-agent failure traces, 14-mode taxonomy | ✓ https://arxiv.org/abs/2503.13657 |
| Who&When | Failure attribution (which agent, which step) | ✓ https://arxiv.org/abs/2505.00212 |
| AgentErrorBench | Root-cause error detection in agent trajectories | ✓ https://github.com/ulab-uiuc/AgentDebug |
| BixBench | Open-ended bioinformatics analysis (K-Dense Analyst) | ✓ named; https://arxiv.org/abs/2503.00096 |
| SWE-bench | Repository-level code repair (AdaptOrch) | https://www.swebench.com |

**Practical implication for the A* version.** SciWeaver's 30-query LLM-judged benchmark matches VMAO's weakest point, so reviewers will discount it. Four additions would make the case:
- Add at least one public agentic benchmark (GAIA or AssistantBench) and one domain benchmark (BixBench).
- Validate the PoT judge against human labels: localisation accuracy on Who&When or MAST-Data.
- Report tokens and latency against Magentic-One, AOP, VeriMAP or VMAO, MaAS and AFlow.
- Ablate "enrich" vs. "rewire" vs. plain retry. VMAO and the self-healing paper both found retries do much of the work.
