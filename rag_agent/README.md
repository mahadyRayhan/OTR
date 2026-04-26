# 🧠 Modular RAG Agent

This agent uses a modular cognitive architecture. Depending on the complexity of the query, you can route the retrieval and synthesis through different thought processes:

1. **CoT (Chain of Thought):** Linear step-by-step reasoning. Good for logical deduction.
2. **ReAct (Reason + Act):** Interleaves reasoning with actions (e.g., retrieving from the Knowledge Base). Good for fact-checking.
3. **ToT (Tree of Thoughts):** Generates multiple hypotheses (branches), evaluates them, and selects the best path. Good for ambiguous questions.
4. **GoT (Graph of Thoughts):** Generates distinct thoughts and merges them together (nodes and edges). Good for synthesis questions requiring multiple parts of a text.

### Self-Reflection (LLM-as-a-Judge)
After generating an answer, the RAG agent passes the `(Question, Context, Final Answer)` to an LLM Judge (Google Gemini). The judge scores the output and returns this metadata to the central router.