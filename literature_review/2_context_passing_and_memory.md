# Literature review: inter-agent context passing, compression and memory, and what it means for SciWeaver's CUGCR

**How this was checked.** I downloaded and read 47 primary sources (46 arXiv PDFs plus one ACL Anthology PDF). For each I pulled the abstract, the Limitations / Future-Work / Conclusion sections and the headline numbers from the paper text. Venues come from the paper header, the arXiv comment field, or an official listing.
- "[venue unverified]" means the venue is from memory or a third-party listing and was not confirmed against an official page.
- Anything else not confirmed in the source is marked [unverified].
- "None stated" means the paper has no explicit limitations section and I found no limitation text in the Conclusion.

The papers are grouped A–F. The three required syntheses (Gaps, Formalizations, Benchmarks/Metrics) come at the end.

---

## A. Compressing context for a single model (the token-level and passage-level toolbox CUGCR builds on)

**1. LLMLingua: Compressing Prompts for Accelerated Inference of LLMs.** Jiang et al., EMNLP 2023. https://arxiv.org/abs/2310.05736
- **Core idea:** Coarse-to-fine compression in three parts:
  - a budget controller that gives different budgets to instruction, demonstrations and question;
  - iterative token-level pruning driven by the perplexity of a small LM;
  - instruction-tuning so the small LM's distribution matches the target LLM.
- **Headline result:** Up to 20x compression with about a 1.5-point drop, tested on GSM8K, BBH, ShareGPT and Arxiv-March23.
- **Limitations:** Performance drops sharply at very high ratios (25–30x on GSM8K). How much a target LLM can recover from a compressed prompt depends on the ratio and on which small LM did the compressing (prompts compressed with GPT2-Alpaca recover less than those compressed with Alpaca-7B).
- **Future work:** None stated beyond hints about compressing the KV cache and handling longer contexts.
- **Relevance:** Its budget controller is the direct ancestor of CUGCR's per-agent token budget. The controller splits a budget across prompt components by type, which is very close to splitting it across CUGCR's atom types (CODE_BLOCK / NAMED_ENTITY / PROSE_CLAIM). Treat it as a baseline that is blind to the downstream task.

**2. LongLLMLingua.** Jiang et al., ACL 2024. https://arxiv.org/abs/2310.06839
- **Core idea:** Question-aware coarse-to-fine compression. It adds document reordering (to counter position bias), dynamic per-document ratios and subsequence recovery.
- **Headline result:**
  - NaturalQuestions: up to +21.4% with about 4x fewer tokens on GPT-3.5-Turbo.
  - LooGLE: 94.0% cost reduction.
  - On ~10k-token prompts at 2–6x compression: 1.4–2.6x lower end-to-end latency.
- **Limitations:**
  - Because compression depends on the question, the same context must be re-compressed for every question, so it cannot be cached. It also doubles LLMLingua's compute. The authors suggest moving from question-aware to task-aware compression to allow reuse.
  - It may weaken when the link between context and question is "more complex and subtle", because its question-awareness is coarse.
- **Relevance:** This is the closest single-LLM analogue to CUGCR's scoring "against the downstream task". Its caching limitation applies directly: in a DAG, CUGCR re-scores the same upstream output separately for each downstream consumer. The reordering result (see Lost in the Middle) suggests CUGCR should also decide the order of the atoms it forwards, not only which ones.

**3. LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression.** Pan et al., Findings of ACL 2024. https://arxiv.org/abs/2403.12968
- **Core idea:** GPT-4 labels which words in MeetingBank can be dropped. An XLM-RoBERTa token classifier is then trained to keep or drop each token.
- **Headline result:**
  - MeetingBank QA at ~3.1x compression: 86.92 EM, against 87.75 for the original prompt and 67.52 for LLMLingua.
  - LongBench at 2k tokens: 39.1, against 34.6 for LLMLingua.
  - 3–6x faster than earlier compressors; 1.6–2.9x end-to-end speedup at 2x–5x compression.
  - Peak GPU memory 2.1 GB, against 16.6 GB for LLMLingua.
- **Limitations:** Trained only on MeetingBank, so generalization is a concern. Adding 50k TriviaQA examples helped only modestly. Why that is so, and whether domains share redundancy patterns, is left for future work.
- **Relevance:** The recipe to copy for a learned CUGCR: distill LLM-judge keep/drop labels into a small atom classifier. The domain-shift limitation matters a lot for bioinformatics code and entity names.

**4. Selective Context: Compressing Context to Enhance Inference Efficiency of LLMs.** Li et al., EMNLP 2023. https://arxiv.org/abs/2310.06201
- **Core idea:** Prune lexical units (spaCy noun phrases) whose self-information under a small LM is low.
- **Headline result:** Halving the context gives 36% less inference memory and 32% less inference time, with only −0.023 BERTScore and −0.038 faithfulness.
- **Limitations:**
  - Depends on phrase-boundary detection; verb phrases are not handled.
  - Uses a percentile threshold, but the best percentile varies by task. A tool to find the optimal threshold is listed as future work.
- **Relevance:** A cheap baseline that is agnostic to the downstream task. Its fixed-threshold problem is the same as choosing CUGCR's budget, which the Re-Planner currently adjusts only reactively.

**5. RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation.** Xu et al., ICLR 2024 [venue unverified]. https://arxiv.org/abs/2310.04408
- **Core idea:** Extractive and abstractive compressors are trained on the end-task signal from a black-box LM. The compressor may output an empty string when the documents do not help ("selective augmentation").
- **Headline result:** Compression to as little as 6% of the original with minimal loss on language modelling and open-domain QA.
- **Limitations:** No explicit Limitations section in v1. The ethics statement notes that training data generated by a commercial LM may contain factual errors. Their manual faithfulness analysis of abstractive summaries found problems, and the paper points to better summarizers as future work.
- **Relevance:** Its key idea is supervising the compressor with the downstream end-task signal, which is what a "causal utility" gate should mean. CUGCR scores by similarity instead. Selective augmentation (forward nothing) is a mode CUGCR should explicitly support.

**6. xRAG: Extreme Context Compression for RAG with One Token.** Cheng et al., NeurIPS 2024. https://arxiv.org/abs/2405.13792
- **Core idea:** A trainable bridge projects the retriever's document embedding into the LLM's input space, so a whole document becomes one soft token.
- **Headline result:** Over 10% average improvement across six knowledge tasks, 3.53x fewer FLOPs, and parity with uncompressed RAG on several benchmarks.
- **Limitations:**
  - Only single dense-vector retrieval; BM25 (sparse) and ColBERT (multi-vector) are untested. The authors say sparse vectors could complement dense ones.
  - Lags RAG by a wide margin on reasoning tasks (HotpotQA, FactKG).
  - Top-1 retrieval only.
  - Future work: reasoning over embeddings, multiple documents, multi-vector retrieval.
- **Relevance:** Independent support for CUGCR's dense+sparse hybrid scoring. It also warns that extreme compression hurts multi-hop reasoning, which is the kind of chaining SciWeaver does (literature → notebook → cloud).

**7. Provence: efficient and robust context pruning for RAG.** Chirkova et al., ICLR 2025 (arXiv comment). https://arxiv.org/abs/2501.16214
- **Core idea:** Context pruning framed as sentence-level sequence labelling, merged into a cross-encoder reranker. The pruning ratio is chosen dynamically for each context.
- **Headline result:** Negligible or no QA loss across domains, and sometimes gains from removing noise. It correctly detects how many sentences are relevant, wherever they sit in the context.
- **Limitations:** QA only; one passage at a time; English only. Future work: other tasks, multi-passage contexts, other languages.
- **Relevance:** Reranking and pruning in a single pass, with an adaptive ratio, is a natural learned replacement for CUGCR's fixed budget. The single-passage limitation matches CUGCR's need to choose atoms across several upstream agents at once.

**8. An Information Bottleneck Perspective for Effective Noise Filtering on RAG.** Zhu et al., ACL 2024. https://arxiv.org/abs/2406.01549
- **Core idea:** Train the filter to maximize I(compression; answer) while minimizing I(compression; retrieved passages). The IB quantity is also used to score filters, select SFT data and build RL rewards.
- **Headline result:** Better answer correctness on QA datasets at a 2.5% compression rate.
- **Limitations:**
  - Needs a capable white-box generator to estimate the mutual-information terms.
  - Trade-off between true-flip rate and false-flip rate; mitigating it with extra training iterations raises training cost.
- **Relevance:** The cleanest single-hop IB objective that could serve as CUGCR's training loss. SciWeaver's specialist agents are mostly API LLMs, so the white-box requirement is a real obstacle.

**9. Lost in the Middle: How Language Models Use Long Contexts.** Liu et al., TACL 2024. https://arxiv.org/abs/2307.03172
- **Core idea:** Controlled multi-document QA and key-value retrieval experiments that move the relevant passage around the context.
- **Headline result:** Accuracy follows a U-shape (primacy and recency bias). With the answer in the middle, GPT-3.5-Turbo does worse than with no documents at all (closed-book 56.1%). Extended-context models are not better at using their context. Reader performance saturates long before retriever recall.
- **Limitations:** No dedicated section; the authors defer other decoding methods to future work.
- **Relevance:** Supports the claim that forwarding less but better-placed context improves the downstream agent. It also shows that CUGCR's output order matters: put high-utility atoms first or last.

---

## B. Communication efficiency in text-based multi-agent systems: topology, edge and message level

**10. AgentPrune ("Cut the Crap: An Economical Communication Pipeline for LLM-based MAS").** Zhang et al., ICLR 2025 [venue unverified; arXiv v1]. https://arxiv.org/abs/2410.02506
- **Core idea:** Formally defines "communication redundancy". Treats the multi-agent system as a spatial-temporal message-passing graph and prunes it once (one-shot, with a learned graph mask).
- **Headline result:**
  - Across six benchmarks, matches state-of-the-art topologies at $5.6 instead of $43.7.
  - Cuts tokens by 28.1–72.8% when plugged into existing frameworks.
  - Robust to two kinds of adversarial attack.
- **Limitations:** None stated (no limitations section).
- **Relevance:** Prunes at the level of whole edges and is fixed after the one-shot pruning. CUGCR works inside an edge, on atoms, at runtime. AgentPrune is a required baseline, and the fact that it is static is your argument for routing per query.

**11. G-Designer: Architecting Multi-agent Communication Topologies via GNNs.** Zhang et al., ICML 2025 (spotlight; ICML listing confirmed via search). https://arxiv.org/abs/2410.11782
- **Core idea:** A variational graph auto-encoder encodes the agents plus a virtual node for the task and decodes a sparse communication topology tailored to that task.
- **Headline result:** MMLU 84.50%; HumanEval pass@1 89.90%; up to 95.33% fewer tokens on HumanEval; only a 0.3% accuracy drop under an agent attack.
- **Limitations:** None stated. The conclusion hopes for future "self-organizing and self-evolving collective intelligence".
- **Relevance:** Plays the same role as SciWeaver's LLM Query Decomposer (building the DAG), but learned. A strong A* paper would need to separate gains from the DAG structure (G-Designer-like) from gains from routing on each edge (CUGCR).

**12. AgentDropout: Dynamic Agent Elimination for Token-Efficient MAS.** Wang et al., ACL 2025 (aclanthology 2025.acl-long.1170 seen in search; not opened). https://arxiv.org/abs/2503.18891
- **Core idea:** Optimizes the adjacency matrix for each communication round, dropping redundant agents and edges.
- **Headline result:** −21.6% prompt tokens, −18.4% completion tokens, +1.14 task performance.
- **Limitations:**
  - Tasks limited to reasoning, math and code.
  - Constrained by predefined roles and prompts.
  - Token use is still "relatively high".
- **Relevance:** Another structural baseline at the agent/edge level.

**13. Optima: Optimizing Effectiveness and Efficiency for LLM-Based MAS.** Chen et al., arXiv (comment: "Under review"); Findings of ACL 2025 [unverified]. https://arxiv.org/abs/2410.08115
- **Core idea:** Iteratively generates, ranks, selects and trains (SFT/DPO, with MCTS-style data generation). The reward combines task success, a token penalty and readability.
- **Headline result:** Up to 2.8x performance with less than 10% of the tokens on information-asymmetric QA (Llama-3-8B / 3.2-3B). It also improves inference-time scaling.
- **Limitations:**
  - Mainly two-agent, same-architecture setups; scaling to 5–10 heterogeneous agents is open.
  - Transfer to very different domains (e.g., QA to code) is untested.
  - Real-world deployment constraints are not addressed.
- **Relevance:** Trains the senders to be concise; CUGCR instead filters at the receiver. The two are complementary, and Optima's readability term is a useful constraint for a scientific copilot.

**14. TodyComm: Task-Oriented Dynamic Communication for Multi-Round LLM MAS.** Fan et al., arXiv 2602.03688 v2 (May 2026; venue not stated). https://arxiv.org/abs/2602.03688
- **Core idea:** Learns, with policy gradient, a topology that changes every round based on agent behaviour, under adversaries and per-node communication budgets. Includes a regret bound.
- **Headline result:** Better task performance on five benchmarks under dynamic adversaries and budget limits, while staying token-efficient.
- **Limitations / future work:** Focused on adversarial settings; extension to collaborative settings is future work.
- **Relevance:** Per-node budget control learned with RL is a principled alternative to CUGCR's hand-set per-agent budgets.

**15. MessageSieve: Runtime Message Pruning for Accurate and Efficient LLM-Based MAS Communication.** You et al., EMNLP 2026 Main (ACL Anthology PDF opened). https://aclanthology.org/2026.emnlp-main.2179/
- **Core idea:** Training-free, no extra LLM call, decides per message instance. Before the prompt is assembled, each incoming message is kept as is, shrunk to compact evidence spans, or dropped, using lightweight "state cards" and local message comparisons. The authors argue that edge- and topology-level pruning is too coarse, since the same edge carries useful evidence in one query and redundancy in another.
- **Headline result:** Beats dense full communication by +4.10 macro points (Qwen3-8B) and +2.59 (Gemini 3.1 Flash-Lite), with 19–49% fewer tokens, across five benchmarks.
- **Limitations:**
  - Fixed graph only; does not cover systems that create agents, tools or links at runtime.
  - Uses deterministic "LITE" views.
  - Two backbones, five benchmarks.
  - Future work: longer horizons and more open-ended workflows.
- **Relevance:** The closest direct competitor to CUGCR, and an A* reviewer will ask for it. CUGCR's novelty claims must be stated relative to it: typed atoms, sparse+dense scoring, budget relaxation coupled to failures, and the ability to rewire the DAG (which MessageSieve explicitly excludes).

**16. RCR-Router: Role-Conditioned Context Routing for Resource-Efficient Multi-Agent LLM Systems.** Liu et al., arXiv 2508.04903 v5 (2 Nov 2026). Listed as Findings of EMNLP 2026 (ACL Anthology 2026.findings-emnlp.425 seen in search; anthology page not opened). https://arxiv.org/abs/2508.04903
- **Core idea:** An execution-time layer that sends role-relevant items from a structured shared memory to each agent under a token budget. It combines role-conditioned relevance scoring, budgeted selection (a scorer trained per item plus an allocator optimized per trajectory) and iterative memory updates.
- **Headline result:** Relative to FullContext, −35% tokens and −38% runtime on average across HotPotQA, ALFWorld and WebShop (plus MuSiQue and 2Wiki), while matching or improving task performance.
- **Limitations:**
  - Does not optimize the topology or roles; assumes a fixed topology and available role or stage metadata.
  - Tested with N = 3–6 agents only; larger systems and longer horizons are uncharacterized.
  - Needs more domains and more backbones.
- **Relevance:** Near-identical in spirit to CUGCR: a budgeted router conditioned on the receiving role. This is the second must-compare baseline. CUGCR's possible differences:
  - typed atom segmentation;
  - a router that reacts to failures (Probing-of-Thought);
  - DAG rewiring.

**17. Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems.** Rajib et al., EMNLP 2026 Main (arXiv comment). https://arxiv.org/abs/2609.00237
- **Core idea:**
  - A learned Memory Write Gate commits only non-redundant reasoning steps.
  - A learned Retrieval Gate gives each agent a compact, relevant subset.
  - An Adaptive Halting Controller stops once memory holds enough evidence.
- **Headline result:**
  - Best average accuracy over five benchmarks (+2.44 points over the strongest baseline) and −31.9% HumanEval cost.
  - Head to head on GSM-Hard: query-only 66.00 accuracy at cost 1.012, full-history 70.27 at 0.985, gated memory 70.55 at 0.587.
- **Limitations:**
  - Closed-domain tasks with verifiable answers only; open-ended tasks would need reward models or human evaluation.
  - Parameter count is used as the compute proxy; wall-clock and FLOPs are measured on only two benchmarks.
  - Open-weight backbones only.
  - Future work: open-ended generation, larger ecosystems, richer rewards than binary success.
- **Relevance:** Shows that learned gates beat both "query-only" and "full history", the two ends CUGCR sits between. The halting controller ("enough evidence") resembles a sufficiency test SciWeaver could add to its judge.

**18. Agent-Radar: Enhancing Multi-Agent Communication through Attention Steering with Context Relevance.** Zhang et al., arXiv 2605.30136 (May 2026). https://arxiv.org/abs/2605.30136
- **Core idea:** Keeps the full transcript but steers each agent's attention (SPA/PASTA backends) toward relevant spans, with temporal and spatial decay. It is soft re-weighting, not hard filtering.
- **Headline result:** Up to +7.64 absolute points over state-of-the-art context-management methods on five benchmarks; stays robust as agents and rounds grow.
- **Limitations:**
  - Needs a threshold and decay factors; adaptive versions are future work.
  - Depends on the quality of its retrieval signal; uncertainty-aware scoring and contradiction detection are suggested.
  - Needs an attention-steering backend, which means white-box models.
- **Relevance:** A "soft routing" alternative to dropping atoms. Not usable with API agents, but useful as an upper bound or ablation.

**19. Chain-of-Agents: LLMs Collaborating on Long-Context Tasks.** Zhang et al., NeurIPS 2024 [venue unverified]. https://arxiv.org/abs/2406.02818
- **Core idea:** Worker agents read chunks one after another and pass "communication units" along; a manager agent synthesizes the result.
- **Headline result:** Up to +10% over RAG, full-context and multi-agent baselines; reduces lost-in-the-middle effects.
- **Limitations:**
  - LLMs aligned to human norms are not optimal for LLM-to-LLM communication; fine-tuning or in-context learning could help.
  - Debate and discussion formats are not explored.
  - Cost and latency could be cut with model routing.
- **Relevance:** An early sequential pipeline that forwards compressed state between agents. Its first limitation is the motivation for both the latent-communication papers and learned routers.

**20. DeLM: Decentralized Multi-Agent Systems with Shared Context.** Mao et al., arXiv 2606.10662 v3 (2026). https://arxiv.org/abs/2606.10662
- **Core idea:** Replaces the central orchestrator with a shared context plus a task queue. Agents claim tasks asynchronously and publish "gists" (with limitations attached) as soon as they are ready; full evidence is fetched on demand.
- **Headline result:** Up to +17.5 accuracy points over the strongest baseline and up to 2.49x faster than its base harness. Evaluated on Terminal-Bench 4.0, DeepSWE v1.1 and SWE-bench Verified, against Codex, Claude Code and AOrchestra.
- **Limitations:**
  - Lower latency and cost per submission can come with higher cost per task.
  - A natural-language shared context is poor at exact aggregation over structured data; code-mediated execution helps.
- **Relevance:** A pull-based alternative ("gist + pointer") to CUGCR's push-based filtering. Losses from over-pruning could be recovered by letting downstream agents dereference a pruned atom, a cheaper fix than relaxing the budget wholesale.

---

## C. Latent and KV-cache communication

**21. KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based MAS.** Ye et al., NeurIPS 2025. https://arxiv.org/abs/2510.12872
- **Core idea:** Reuses the KV cache of shared text across agents whose prefixes differ. Offsets are corrected using an online pool of "anchors" that record how the cache deviates under different prefixes. Training-free.
- **Headline result:** Over 70% cache reuse; about 6.7x average prefill speedup with three agents on one H100; up to 7.8x with five agents. Workloads: RAG, math and coding.
- **Limitations:**
  - Text only; decoding latency is not reduced.
  - Applies directly only to homogeneous agents.
  - Fully dynamic, unstructured (debate-style) cases are not covered.
  - Heterogeneous attention formulations are unexplored.
- **Relevance:** Mostly about speed, not about what to send. Shows that once CUGCR has picked atoms, their KV states could be reused when SciWeaver agents share a backbone.

**22. Cache-to-Cache (C2C): Direct Semantic Communication Between LLMs.** Fu et al., ICLR 2026. https://arxiv.org/abs/2510.03215
- **Core idea:** A trained fuser projects the sender's KV cache into the receiver's; a learnable gate chooses which receiver layers take it in.
- **Headline result:** +6.4–14.2% average accuracy over the individual models, +3.1–5.4% over text-to-text communication, and 2.5x lower latency.
- **Limitations / future work:** Pairwise only. Communication among many models is future work; the appendix sketches O(N) fusers through a shared "KV-Latent" space.
- **Relevance:** Strong evidence that text handoffs lose information, but it needs white-box models and paired training. Useful for the "what is lost when we route text" argument.

**23. LatentMAS: Latent Collaboration in Multi-Agent Systems.** Zou et al., ICML 2026 Spotlight. https://arxiv.org/abs/2511.20639
- **Core idea:** Agents "think" in last-layer hidden states and hand off a shared latent working memory, which the authors claim is a lossless transfer. Training-free.
- **Headline result:** Up to +14.6% accuracy, 70.8–83.7% fewer output tokens and 4–4.3x faster inference across 9 benchmarks.
- **Limitations:**
  - Assumes all agents share the same transformer shape; adapters for heterogeneous agents are suggested.
  - Adapting post-training methods to latent protocols is future work.
  - The authors added a text "debug mode" because auditing latent traffic is otherwise hard.
- **Relevance:** Lossless transfer to the next agent is the opposite of CUGCR's lossy filtering. LatentMAS can reach that because its agents share one architecture; SciWeaver's API-based, heterogeneous agents cannot.

**24. Thought Communication in Multiagent Collaboration.** Zheng et al., NeurIPS 2025 Spotlight. https://arxiv.org/abs/2510.20733
- **Core idea:** Treats agent states as an unknown function of latent thoughts and proves, nonparametrically, that shared and private thoughts between any pair of agents, and the overall sharing structure, are identifiable. A sparsity-regularized autoencoder extracts the thoughts and sends each agent only the relevant ones.
- **Headline result:** Qwen3-1.7B reaches 93% on MATH, +17.2 points absolute over Multiagent Finetuning; on average +19.06% relative over that state-of-the-art method. Also better consensus.
- **Limitations:** Needs model states, so not feasible for closed models; context-aware embeddings are suggested as a substitute. Other modalities are untested.
- **Relevance:** The only identifiability result on *which* information should go to *which* agent. Conceptually this is CUGCR's question ("relevant thoughts per agent") posed at the latent level, and it is the strongest theoretical citation for the "what must cross an edge" framing.

**25. Interlat: Enabling Agents to Communicate Entirely in Latent Space.** Du et al., ACL 2026 (arXiv comment). https://arxiv.org/abs/2511.09149
- **Core idea:** Agents send last hidden states, and a learned latent-reasoning compressor shortens the message further.
- **Headline result:** Beats fine-tuned CoT and single agents, including across heterogeneous models; compression gives up to 24x faster inference (abstract), or up to 24x less communication (intro).
- **Limitations:**
  - Two agents, ALFWorld and MATH only; larger teams, tools and memory untested.
  - Needs hidden-state access, which closed or API models do not give.
- **Relevance:** Its learned message compression is the latent counterpart of a learned CUGCR.

**26. SelectKV and HyLaMAS: Selective and Hybrid-State Communication for Latent MAS.** Liao et al., NeurIPS 2026 Workshop RAAAI (arXiv comment). https://arxiv.org/abs/2610.37779
- **Core idea:** SelectKV chooses which newly generated KV positions cross an agent boundary while protecting the inherited context. HyLaMAS passes KV plus recurrent states for hybrid linear-attention models.
- **Headline result:**
  - SelectKV: 9.7–9.9% less KV payload. Accuracy unchanged on 2 of 6 benchmarks, within 1 point on 3, and −6 points on HumanEval+.
  - HyLaMAS: 53–80% fewer decoded tokens and 1.16–2.78x speedups.
- **Limitations:**
  - 100 examples per benchmark and one model scale.
  - The two methods are evaluated separately.
  - Smaller logical payload does not imply real network savings, and selection itself costs compute.
  - Future work: adaptive communication policies.
- **Relevance:** Selective forwarding in latent space hits the same problem as CUGCR, a sharp drop on code. CUGCR's treatment of CODE_BLOCK atoms deserves a dedicated evaluation.

**27. Q-KVComm: Efficient Multi-Agent Communication via Adaptive KV Cache Compression.** Kriuk & Ng, arXiv 2512.17914 (7-page preprint). https://arxiv.org/abs/2512.17914
- **Core idea:** Layer-wise quantized KV transfer, plus "hybrid information extraction" (keywords, named-entity recognition, domain patterns) to make sure key facts survive.
- **Headline result:** 5–6x compression with coherence above 0.77, tested only on 1.1–1.5B models.
- **Limitations:** None stated; the evaluation is small.
- **Relevance:** A weak paper, but it independently uses entity-based "fact atoms" to protect key content, which supports CUGCR's NAMED_ENTITY atom type.

**28. Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication.** Zhang & Emu, arXiv 2607.26773 (2026). https://arxiv.org/abs/2607.26773
- **Core idea:** Replaces messages in controlled ways at the point where the sender's output enters the receiver. Four message conditions yield five measures:
  - how much sender information is encoded;
  - whether the receiver is sensitive to the message's presence;
  - whether it is sensitive to the message's identity;
  - the task value of content specific to this example;
  - the added value of a separate agent.
- **Headline result:** On GSM8K with Qwen3-4B, an overall −1.00-point effect splits into −6.17 (effect also produced by a message from another example) and +5.17 (content specific to this example). Both components reverse at 8B. Aggregate accuracy therefore cannot identify the mechanism.
- **Limitations:** None stated beyond scope (latent relay, Qwen3, three tasks).
- **Relevance:** Directly reusable as an evaluation protocol for CUGCR. Swap the routed atoms for atoms from another query or for random atoms, and show that the downstream agent actually *uses* the routed, example-specific atoms. This is what would justify the "Causal" in CUGCR.

**29. Why do AI agents communicate in human language?** Zhou et al., arXiv 2506.02739 (position paper). https://arxiv.org/abs/2506.02739
- **Core idea:** Argues that natural language does not match the vector spaces LLMs operate in, which causes information loss and behavioural drift. Also argues that LLMs lack mechanisms for role continuity, task boundaries and dependencies between agents.
- **Headline result:** None (position paper).
- **Limitations:** Not empirical.
- **Relevance:** A framing citation only.

---

## D. Memory and context engineering

**30. MemGPT: Towards LLMs as Operating Systems.** Packer et al., arXiv 2310.08560 (no venue in arXiv metadata). https://arxiv.org/abs/2310.08560
- **Core idea:** Virtual context management: the LLM pages content between main context and external storage through function calls, triggered by interrupts.
- **Headline result:** Handles documents far beyond the context window and multi-session chat. Results depend on the backbone: weaker GPT-3.5 function calling hurts [specific numbers not extracted].
- **Limitations:** No dedicated section. Future work: other unbounded-context domains, other memory-tier technologies (databases, caches), better control-flow and memory-management policies.
- **Relevance:** A self-directed paging model; CUGCR is router-directed. A hybrid in which a downstream agent can page in a pruned atom would mitigate over-pruning (compare DeLM).

**31. Generative Agents: Interactive Simulacra of Human Behavior.** Park et al., UIST 2023. https://arxiv.org/abs/2304.03442
- **Core idea:** A memory stream, reflection and planning. Retrieval scores memories by a weighted mix of recency, importance and relevance.
- **Headline result:** Believable behaviour in a 25-agent sandbox; ablations show that memory, reflection and planning each matter.
- **Limitations / future work:**
  - Tune the relevance, recency and importance functions.
  - Cost: thousands of dollars to simulate 25 agents for two days.
  - Short evaluation horizon, and the crowdworker baseline is not the best human performance.
- **Relevance:** Its scoring rule has the same shape as CUGCR's atom score. Adding *importance* and *recency* terms to relevance is a cheap extension.

**32. A-MEM: Agentic Memory for LLM Agents.** Xu et al., NeurIPS 2025 (arXiv comment). https://arxiv.org/abs/2502.12110
- **Core idea:** Zettelkasten-style notes with descriptions, keywords and tags; links are created dynamically; new memories can update old ones ("memory evolution").
- **Headline result:** Better than state-of-the-art baselines on long-term conversational QA across six foundation models [specific numbers not extracted].
- **Limitations:** Organization quality depends on the LLM, which may make different links; text only, with multimodal extension as future work.
- **Relevance:** Typed, attributed notes resemble typed atoms. Linking atoms across agents (for example an entity from the literature agent to a notebook) would let CUGCR forward *linked bundles* instead of isolated atoms.

**33. Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory.** Chhikara et al., arXiv 2504.19413 (venue not in metadata). https://arxiv.org/abs/2504.19413
- **Core idea:** Extracts, consolidates (ADD / UPDATE / DELETE) and retrieves compact memories; Mem0^g adds a graph.
- **Headline result:** On LoCoMo, +26% relative LLM-as-a-Judge score over OpenAI memory, 91% lower p95 latency and more than 90% token-cost saving relative to full context.
- **Limitations / future work:** Optimize graph operations, which add latency; hierarchical memory architectures; more sophisticated reasoning over memory [rest of the sentence truncated in extraction].
- **Relevance:** Its consolidation operators could dedupe atoms that several upstream agents emit, before CUGCR forwards them.

**34. MemoryOS: Memory OS of AI Agent.** Kang et al., EMNLP 2025 (ACL Anthology listing via search). https://arxiv.org/abs/2506.06326
- **Core idea:** Short-, mid- and long-term (persona) memory tiers, with FIFO and segmented-page updates and heat-based eviction.
- **Headline result:** LoCoMo with GPT-4o-mini: +49.11% F1 and +46.18% BLEU-1 over baselines.
- **Limitations:** None stated.
- **Relevance:** Its persona module is the closest analogue to SciWeaver's User Preference Profiler (the Procedural/Analytical/Conceptual intent vector). An A* version could condition CUGCR's budget on that intent vector.

**35. G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems.** Zhang et al., NeurIPS 2025 [venue unverified]. https://arxiv.org/abs/2506.07398
- **Core idea:** Three-tier graph memory for multi-agent systems (insight, query and interaction graphs). Retrieval runs both ways, returning cross-trial insights and condensed collaboration trajectories customized per agent.
- **Headline result:** Up to +20.89% on embodied action tasks (search snippet also gives +10.12% on knowledge QA [unverified]).
- **Limitations:** More diverse tasks (e.g., medical QA) are future work.
- **Relevance:** Memory across queries of which collaboration patterns and routings worked. CUGCR could warm-start budgets from the routings that succeeded on similar past queries.

**36. ACON: Optimizing Context Compression for Long-horizon LLM Agents.** Kang et al., ICML 2026 (arXiv comment). https://arxiv.org/abs/2510.00615
- **Core idea:** Optimizes compression guidelines in natural language. It contrasts trajectories that succeed with full context but fail with compressed context, refines the guideline, then distils the compressor into small models.
- **Headline result:** 26–54% lower peak tokens with better task success than compression baselines; lets Qwen3-14B approach larger models. Benchmarks: AppWorld, OfficeBench, multi-objective QA.
- **Limitations:**
  - Mainly GPT models; Gemini, Claude and very large open models untested.
  - Controlled benchmarks, not real deployments.
  - The compressor adds API cost.
- **Relevance:** Very close to SciWeaver's Probing-of-Thought loop: classify the failure, then fix the context. ACON turns failure analysis into a *learned* compression policy. That suggests SciWeaver's Re-Planner should *update CUGCR's scoring rules*, not only relax budgets case by case.

**37. ACE: Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models.** Zhang et al., ICLR 2026. https://arxiv.org/abs/2510.04618
- **Core idea:** Treats the context as a playbook updated incrementally (generate, reflect, curate). Names two failure modes of rewriting contexts: *brevity bias* and *context collapse* (in their example, 18,282 tokens collapse to 122 and accuracy falls from 66.7 to 57.1).
- **Headline result:** +10.6% on agent benchmarks, +8.6% on finance; matches the top production agent on the AppWorld leaderboard on overall average.
- **Limitations:** Depends on a strong Reflector, without which the context gets noisy. Some tasks (HotPotQA, Game of 24) need only short contexts.
- **Relevance:** Brevity bias and context collapse are the risks of aggressive CUGCR budgets. Measuring them explicitly would strengthen the paper.

**38. A Survey of Context Engineering for LLMs.** Mei et al., arXiv 2507.13334 (166 pages, "ongoing work"). https://arxiv.org/abs/2507.13334
- **Core idea:** A taxonomy:
  - components: retrieval/generation, processing, management (memory, compression);
  - implementations: RAG, memory, tool use, multi-agent systems.
- **Headline result:** Identifies a comprehension–generation asymmetry as the key gap.
- **Limitations / future directions:**
  - Theoretical foundations and unified frameworks, including mathematical frameworks to predict coordination effectiveness in multi-agent orchestration.
  - Evaluation frameworks for dynamic, multi-component systems.
  - Intelligent context assembly.
- **Relevance:** Use it for positioning: CUGCR is "context management for multi-agent orchestration".

---

## E. Theory and measurement of information loss between agents

**39. When Do Multi-Agent Systems Help? An Information Bottleneck Perspective.** Yu, Zhou et al. (Texas A&M; Ji, Caverlee, Qian), arXiv 2607.16133 (Jul 2026). https://arxiv.org/abs/2607.16133
- **Core idea:** A single agent keeps the whole trace in one context, while a multi-agent system uses isolated contexts joined by bounded relays. With unlimited relay bandwidth, a multi-agent system can simulate any single agent. Any real advantage therefore comes from *bounded* relays, formalized as an information bottleneck with a capability-dependent β: gain = context reduction minus relay information loss Δ.
- **Headline result:**
  - 18 controlled experiments, five benchmarks, three model scales.
  - Multi-agent systems help when relays are near-sufficient, especially for weaker models.
  - Gains shrink or reverse when relays lose information, especially for stronger models.
  - Causal test: deliberately removing information the downstream agent needs from the relay (location and object state in ALFWorld; price and attributes in WebShop) consistently lowers performance.
- **Limitations:**
  - β is qualitative, not estimated.
  - δ is assigned by task-structure analysis, not measured.
  - Future work: quantitative estimators for both, used to train adaptive relays; jointly optimize the relay encoder with the downstream task; adaptive relays for heterogeneous teams.
- **Relevance:** The single most relevant theory paper for CUGCR. It formalizes one DAG edge as an IB relay and says outright that the next step is a learned, adaptive relay. CUGCR (with the Re-Planner relaxing the budget) is a heuristic instance of exactly that. It also predicts that CUGCR's benefit should shrink with stronger backbones, which SciWeaver should test.

**40. An Information Theoretic Perspective on Agentic System Design.** He, Narayan et al. (Stanford), arXiv 2512.21720. https://arxiv.org/abs/2512.21720
- **Core idea:** Views compressor→predictor pipelines (Deep Research, Claude Code subagents) as a noisy channel. Proposes a simple, unbiased estimator of the mutual information between the context and its compression that inference servers can compute, as a task-agnostic quality proxy.
- **Headline result:**
  - Mutual information strongly predicts downstream performance (the correlation figure in the paper appears to be with perplexity, r = −0.84 [verify]).
  - A 7B compressor is 1.6x more accurate, 4.6x more concise and carries 5.5x more bits per token than the 1.5B version.
  - Scaling the compressor beats scaling the predictor.
  - 3B local compressors recover 99% of frontier accuracy at 26% of API cost.
- **Limitations:**
  - At 1–3B scale the estimator relies on proxy models and log-probs, which adds variance and bias.
  - Focus on non-reasoning models with single-round communication; iterative multi-agent workflows are not covered.
  - Future work: InfoNCE estimators, using information theory to route compressors and decide fallbacks, rate-distortion training objectives.
- **Relevance:** Gives SciWeaver a *task-agnostic metric of information loss on each edge*: the MI between an upstream output and its CUGCR-filtered version, in bits per token. Its multi-round limitation is exactly SciWeaver's DAG setting.

**41. Faithful, Not Corrective: Model Capability Governs Message-Format Effects in Multi-Hop Agent Relays.** Zeng, arXiv 2607.09678 v2 (Sep 2026; single independent author). https://arxiv.org/abs/2607.09678
- **Core idea:** Briefs of 12 programmatic atomic facts are relayed over 6 hops in five formats (free NL, precision-instructed NL, JSON, triples, key–value) at two capability tiers, with errors injected deliberately.
- **Headline result:**
  - A strong relay is nearly lossless in every format (hop-6 QA recall ≥ 0.973), with loss concentrated at the first encoding step.
  - A weak 1.5B relay spreads hop-6 recall across formats 8.7x more widely.
  - Injected errors persist in 83–100% of chains: fixed-key schemas transmit faithfully but do not correct errors.
- **Limitations:**
  - Synthetic, templated, English corpus.
  - Only two capability points, which differ in several ways at once.
  - Preservation is the instructed objective, so pipelines that compress on purpose face a different objective.
  - Injected errors are independent leaf facts only.
- **Relevance:** A ready-made protocol for measuring information loss across hops. It suggests serializing CUGCR's typed atoms in a key–value or JSON schema, which matters most when cheap models sit in relay positions.

**42. Facts Without Rules: Boundary Metadata Collapse in Multi-Agent LLM Handoffs.** Wang et al., arXiv 2608.29028 (Aug 2026). https://arxiv.org/abs/2608.29028
- **Core idea:** Compressing a handoff keeps operational facts but strips the "boundary metadata" that governs how those facts may be used ("summary collapse"). Measured as marker survival σ_b (human-validated judge, κ = 0.74) against operational-fact survival σ_op.
- **Headline result:**
  - σ_b and σ_op are almost uncorrelated (Pearson r ≈ 0).
  - Free-text handoffs keep σ_b ≈ 0.80; a 25-word budget drops it to ≈ 0.57 while σ_op stays near ceiling.
  - Vague boundaries leak in 73% of GPT cases and 50% of DeepSeek cases; explicit boundaries leak in under 15%.
  - A gold typed audience-allowlist schema brings leakage to 0/48.
- **Limitations:**
  - Synthetic transcripts and simulated audiences.
  - Independent annotation only for experiment E7.
  - English only.
  - Open problem: automatically extracting and validating typed boundaries.
- **Relevance:** A critical result for CUGCR. Relevance-based atom scoring will keep entities and claims and drop *qualifiers* ("only for human data", "requires a GPU quota", licence or provenance caveats). Two fixes follow:
  - add a CONSTRAINT/METADATA atom type that is never dropped;
  - evaluate atom-type survival separately for each type.

---

## F. Attribution, provenance and failure analysis of information flowing between agents

**43. ContextCite: Attributing Model Generation to Context.** Cohen-Wang et al., NeurIPS 2024 [venue unverified]. https://arxiv.org/abs/2409.00729
- **Core idea:** Fits a sparse linear surrogate over random ablations of context sources to attribute each generated statement to the sources behind it. Applications: verification, pruning the context to improve answers, detecting poisoning.
- **Headline result:** A faithful surrogate, evaluated with LDS and top-k log-prob drop; pruning the context to the top attributed sources improves QA [specific numbers not extracted].
- **Limitations:**
  - The linear assumption may fail when many sources carry redundant copies of the same information.
  - Statements that follow from earlier generated statements are not attributed to the context.
- **Relevance:** A ready offline labeller. Run ContextCite on downstream outputs over the *forwarded atoms* to get counterfactual utility labels, then train CUGCR's scorer on them. This gives "causal utility" a defensible meaning. Its redundancy failure mode applies when several upstream agents send the same entity.

**44. MAST: Why Do Multi-Agent LLM Systems Fail?** Cemri, Pan et al., NeurIPS 2025 Datasets & Benchmarks. https://arxiv.org/abs/2503.13657
- **Core idea:** Over 1,600 annotated traces from 7 frameworks, organized into a taxonomy of 14 failure modes in 3 categories (κ = 0.88), plus an LLM-as-judge annotator.
- **Headline result:** Inter-agent misalignment failure modes include:
  - FM-2.4 information withholding (0.85%);
  - FM-2.5 ignored other agent's input (1.90%);
  - FM-2.6 reasoning–action mismatch (13.2%).
  - Separately, FM-1.4 context mismanagement is a system-design failure that can look the same.
  - The authors note that "missing information" on the surface can come from different root causes.
- **Limitations:** None stated beyond acknowledging that some failures come from fundamental LLM limits.
- **Relevance:** SciWeaver's Probing-of-Thought failure-type classifier should adopt or map to MAST categories, especially to tell CUGCR-induced withholding (a routing error) apart from "ignored input" (a receiver error).

**45. Which Agent Causes Task Failures and When? (Who&When).** Zhang et al., ICML 2025. https://arxiv.org/abs/2505.00212
- **Core idea:** Defines automated failure attribution (which agent, which step) and releases a dataset of logs from 127 multi-agent systems.
- **Headline result:** The best method is 53.5% accurate on the responsible agent but only 14.2% on the responsible step; o1 and R1 are not practically usable.
- **Limitations:** The task is hard and needs more research.
- **Relevance:** A warning for Probing-of-Thought: a per-node LLM judge deciding *which node* failed, and *why*, is unreliable by this benchmark. SciWeaver should report how accurately its judge localizes failures, perhaps on Who&When.

**46. C3 / "The Trace Is the State: Exact Credit Assignment for LLM Agent Teams."** Chen et al., arXiv 2603.06859 v4 (title changed in v3; under review). https://arxiv.org/abs/2603.06859
- **Core idea:** When agents communicate only through a logged shared context, "the trace is the state", so the true counterfactual can be executed: swap one message at a decision point, continue the run to the terminal reward, and compare. The credit is unbiased up to Monte Carlo error, with a derived variance law that does not depend on the number of agents.
- **Headline result:** With 4 continuations, 0.69 rank correlation against a 16-continuation reference, close to the 0.73 agreement between two references; a learned critic does worse. Trains better teams with fewer tokens.
- **Limitations:** Applies only where the trace fully captures what each role reads. The authors prescribe logging everything each role reads.
- **Relevance:** SciWeaver's DAG with logged agent outputs satisfies "trace is state". Atom-level C3 (swap or drop atom k on edge e, re-run the downstream node) is the *exact* definition of CUGCR's causal utility, and a gold standard to calibrate the cheap similarity score against.

**47. From Agent Traces to Trust: A Survey of Evidence Tracing and Execution Provenance in LLM Agents.** Wang et al., arXiv 2606.04990 v6 (2026). https://arxiv.org/abs/2606.04990
- **Core idea:** Unifies execution provenance (a typed graph of an agent run) and evidence tracing (its projection onto evidence–support relations), with a six-dimension taxonomy.
- **Headline result:** No system in the survey explains how evidence, tools, memory, observations, *messages* and actions together shape the final answer. No benchmark evaluates the full provenance chain.
- **Limitations / future work:**
  - Unified, interoperable trace schemas.
  - Multi-agent provenance benchmarks covering evidence propagation, errors introduced in communication, dependency between agents and responsibility attribution.
- **Relevance:** CUGCR's typed atoms are natural provenance units. Each forwarded atom could carry its source agent, document or notebook ID, and score. That would make SciWeaver one of the few systems with provenance at the message level, and it is a gap this survey explicitly names.

---

## 1. Gaps: open problems in inter-agent context passing (late 2026)

1. **Relevance is not utility.** Every runtime router uses relevance or similarity proxies, including RCR-Router, MessageSieve, Agent-Radar, Gated-Memory's retrieval gate and SciWeaver's CUGCR. True interventional utility is only measured offline and expensively:
   - executed counterfactuals in C3;
   - controlled message replacement in the causal audit;
   - ablation surrogates in ContextCite.
   
   No method yet learns a cheap runtime estimator of an atom's causal utility for the downstream agent that is calibrated against executed counterfactuals.
2. **Allocating the budget jointly across a DAG.** Every method budgets per agent or per edge (RCR, CUGCR, TodyComm's per-node control), or prunes edges structurally (AgentPrune, G-Designer, AgentDropout). None handles atoms that matter two or more hops downstream, or how losses compound across a chain. Two papers point toward this without solving it:
   - IB-MAS explicitly calls for jointly optimizing the relay encoder with the downstream task;
   - Faithful-Not-Corrective shows that errors persist across hops.
3. **No standard measure of information loss.** What exists is limited:
   - the MI estimator of He et al. is single-round;
   - IB-MAS's Δ and β are qualitative or hand-assigned;
   - relay-fidelity testbeds are synthetic (12 templated facts);
   - Facts-Without-Rules shows that fact-survival metrics miss the loss of constraints and metadata entirely.
   
   No benchmark measures, edge by edge, how much downstream-required information was lost in a realistic heterogeneous scientific workflow.
4. **Content that is not prose.** Compressors are trained or tested on prose and QA: LLMLingua-2 on MeetingBank, Provence on English QA. Latent selective transfer drops 6 points on code (SelectKV on HumanEval+). Compressing typed content (code, entity IDs, numbers, licences and caveats) while keeping its meaning is unstudied. This is CUGCR's natural niche.
5. **Agents that are heterogeneous or closed-source.** All the strongest lossless methods require white-box agents with the same architecture: LatentMAS, C2C, KVCOMM, ThoughtComm, Interlat, Agent-Radar. Systems mixing API LLMs, tools and retrievers, like SciWeaver, still communicate in text, and there is no theory of the best *text* relay for them.
6. **Failure-driven adaptation of routing, and reliable blame.** ACON and ACE learn compression or context rules from failures. Yet localizing a failure is unreliable: Who&When reaches 14.2% at the step level, and MAST notes that "missing information" has several possible root causes. Telling "the router dropped it" apart from "the receiver ignored it" (FM-2.4 against FM-2.5) is unsolved, and a re-planner needs that distinction to fix the right component.
7. **Provenance through compression.** Summaries and filtered messages lose their links to sources, and the provenance survey finds no system that explains how messages jointly shape outputs. Keeping evidence pointers through each routing step, and verifying them, is open (compare DeLM's gist-plus-pointer design).
8. **Budgets that depend on capability.** IB-MAS (β) and Faithful-Not-Corrective (tier-dependence) both show that the best amount of compression depends on the receiving model. No router conditions its budget on the receiver's capability or on its uncertainty.

## 2. Has anyone formalized "what minimal context must cross a DAG edge"?

**Closest works:**
- **IB-MAS (2607.16133).** The most direct formalization. Each relay is an information bottleneck, min I(X; M) − β·I(M; Y_downstream), with a proven equivalence between a single agent and a multi-agent system under unlimited bandwidth, and a causal test showing that removing required information hurts. It does *not* estimate β or δ, gives no learned relay, and does not model multi-hop DAGs.
- **IB noise filtering for RAG (2406.01549).** An IB training objective for a single-hop filter; needs a white-box generator.
- **Information-theoretic agentic design (2512.21720).** A practical MI estimator for compressor→predictor channels; single round, task-agnostic, so it does not tell you which bits the downstream task needs.
- **Thought Communication (2510.20733).** Identifiability of shared and private latent thoughts and of who should receive what. It is the only *theoretical* answer to "which information goes to which agent", but it works in latent space with white-box states.
- **C3 (2603.06859) and the causal audit (2607.26773).** Executable counterfactual, interventional definitions of a message's value, and a decomposition of message effects. These give the right *causal estimand*, but they are evaluation and credit tools, not routers.
- **Learned or heuristic routers:**
  - RCR-Router: role-conditioned scorer with a trajectory-level allocator;
  - Gated-Memory Routing: learned write and retrieval gates plus sufficiency-based halting;
  - MessageSieve: keep, compact or omit each message instance;
  - RECOMP: compressor trained on the end task that may output nothing;
  - ACON: compression guidelines optimized from failures.

**Still missing:**
- A definition of the *minimal sufficient* edge context on a DAG, i.e. the smallest Z_e with P(success | Z_e, other inputs) ≥ P(success | full) − ε, that handles multi-hop dependencies.
- A runtime estimator of that quantity calibrated against executed counterfactuals.
- Guarantees on sufficiency or regret under a token budget (TodyComm has a regret bound, but for topology, not content).
- Treatment of typed and constraint atoms.
- Any of this for heterogeneous black-box agents.

A formulation that combines the IB-MAS objective, C3-style counterfactual labels and typed atoms with a protected constraint class would be a novel, defensible A*-level contribution for CUGCR.

## 3. Benchmarks and metrics used for communication efficiency and information loss

**Efficiency metrics**
- Prompt and completion tokens; compression ratio (LLMLingua family, RECOMP 6%, IB-RAG 2.5%).
- Dollar cost (AgentPrune $5.6 vs $43.7).
- Peak tokens (ACON).
- Runtime (RCR −38%).
- TTFT and prefill speedup (KVCOMM 6.7–7.8x) and cache reuse rate (KVCOMM over 70%).
- End-to-end latency (LongLLMLingua 1.4–2.6x; C2C 2.5x; LatentMAS 4–4.3x).
- FLOPs (xRAG 3.53x).
- p95 latency (Mem0).
- GPU memory (LLMLingua-2; Selective Context).
- Parameter-count compute proxy (Gated-Memory).
- Logical KV payload (SelectKV).

**Information-loss and fidelity metrics**
- Mutual information between context and compression, and bits per token (He et al.).
- Relay information loss Δ and the β trade-off (IB-MAS).
- Hop-k QA recall and judge-free fact recall; injected-error persistence (Faithful-Not-Corrective).
- Boundary-marker survival σ_b against operational-fact survival σ_op; leakage rate (Facts-Without-Rules).
- Five causal-audit measures: encoded sender information, sensitivity to presence and identity, value of example-specific content, value of another agent.
- Counterfactual credit with rank correlation to a reference (C3).
- ContextCite LDS and top-k log-prob drop.
- Faithfulness, BERTScore, ROUGE (Selective Context); faithfulness and comprehensiveness of summaries (RECOMP manual evaluation).
- Consensus rate (ThoughtComm).
- Coherence and relevance scores (Q-KVComm).
- MAST failure-mode rates; Who&When agent-level and step-level accuracy.

**Task benchmarks used**
- Topology and pruning papers: MMLU, GSM8K, MATH, MultiArith, SVAMP, AQuA, HumanEval / HumanEval+.
- Information-asymmetric and multi-hop QA: HotpotQA, 2WikiMultiHop, MuSiQue, TriviaQA, CBT (Optima, RCR).
- Long-context and RAG: NaturalQuestions multi-document, LongBench, ZeroSCROLLS, LooGLE, MeetingBank; FactKG and others (xRAG).
- Embodied and web: ALFWorld, WebShop (RCR, IB-MAS, Interlat).
- Long-horizon agents: AppWorld, OfficeBench (ACON, ACE).
- Software engineering: Terminal-Bench 4.0, DeepSWE, SWE-bench Verified, ProgramBench (DeLM).
- Memory: LoCoMo (Mem0, MemoryOS, A-MEM).
- Math reasoning: GSM-Hard, ARC-C, MATH-500 (Gated-Memory, causal audit).

**What this means for SciWeaver's evaluation:**
- 30 queries scored by an LLM judge falls well below the standard of these papers.
- At minimum, add:
  - a public multi-hop or agentic benchmark (HotpotQA/MuSiQue, plus ALFWorld or AppWorld);
  - token, cost and latency curves;
  - an information-loss audit for each edge, for example MI bits per token (He et al.) and per-type atom survival including constraint atoms (Facts-Without-Rules);
  - a causal-audit ablation that swaps in atoms from another query and random atoms (2607.26773);
  - head-to-head baselines: RCR-Router, MessageSieve, AgentPrune, and LLMLingua-2 / LongLLMLingua applied to each edge.

---

**Verification notes**
- **Venues not confirmed against official pages:** RECOMP (ICLR 2024), AgentPrune (ICLR 2025), Optima (Findings ACL 2025; arXiv says "Under review"), Chain-of-Agents (NeurIPS 2024), ContextCite (NeurIPS 2024), G-Memory (NeurIPS 2025).
- **Venues seen only in search listings:** AgentDropout (ACL 2025) and RCR-Router (Findings EMNLP 2026), from ACL Anthology URLs; MemoryOS (EMNLP 2025), from the anthology listing via search.
- **Numbers not extracted** where marked "[not extracted]" or "[unverified]".
- **No venue yet (arXiv preprints):** IB-MAS, DeLM, Agent-Radar, Facts-Without-Rules, Faithful-Not-Corrective, the causal audit, TodyComm, C3 and the provenance survey. Peer review may change their claims.
- **Source files:** all PDFs and extracted texts are in `/private/tmp/claude-502/-Users-mhr6wb-Library-CloudStorage-OneDrive-UniversityofMissouri-Projects-OTR/6b04ea84-22d4-49f1-aaf9-a58a52e2c8a7/scratchpad/papers/`.
