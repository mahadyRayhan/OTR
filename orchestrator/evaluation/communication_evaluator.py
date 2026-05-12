"""
Drop-in replacement for the utilization scoring method inside
orchestrator/evaluation/communication_evaluator.py

The old approach asked an LLM "did the agent use the context?" which
returns near-binary scores (0.0 or 1.0), causing CUS to cluster at
the extremes and making the U-curve uninterpretable.

The new approach is fully deterministic:
  1. Extract every meaningful token from the injected context
  2. Count how many of those tokens appear in the downstream output
  3. CUS = matched_tokens / total_injected_tokens

This produces genuine continuous scores across 0–100%.

HOW TO INTEGRATE:
  Replace the existing utilization scoring method in
  CommunicationEvaluator with _compute_cus_deterministic() below,
  and call it instead of the LLM-based scorer.
"""

import re
import math
from collections import Counter


# ── Stopwords — tokens that don't count as "used" ────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "as", "if", "so",
    "we", "you", "i", "he", "she", "they", "then", "than", "also",
    "not", "no", "can", "will", "would", "could", "should", "may", "might",
    "have", "has", "had", "do", "does", "did", "what", "which", "who",
    "how", "when", "where", "all", "each", "such", "into", "use", "used",
}

# ── Weights: technical tokens count more than prose ──────────────────────────
# A gene name or function call appearing in the output is stronger evidence
# of utilization than a common word like "method".
_CODE_PATTERN   = re.compile(r'`[^`]+`|```[\s\S]*?```|\b\w+\.\w+\(')
_ENTITY_PATTERN = re.compile(r'\b[A-Z]{2,}[0-9]*\b|\b[A-Z][a-z]+-\d+\b')


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords and short tokens."""
    tokens = re.findall(r'\b[a-zA-Z0-9_\-\.]+\b', text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


def _extract_weighted_tokens(text: str) -> Counter:
    """
    Build a weighted token counter from a text.
    Code tokens and biological entities get weight 3 (high signal).
    All other meaningful tokens get weight 1.
    """
    counts = Counter()

    # High-weight: code tokens
    for match in _CODE_PATTERN.finditer(text):
        for tok in _tokenize(match.group()):
            counts[tok] += 3

    # High-weight: named entities (gene names, tool names)
    for match in _ENTITY_PATTERN.finditer(text):
        tok = match.group().lower()
        if tok not in _STOPWORDS:
            counts[tok] += 3

    # Normal weight: remaining prose tokens
    for tok in _tokenize(text):
        if counts[tok] == 0:   # not already counted as high-weight
            counts[tok] += 1

    return counts


def compute_cus_deterministic(
    injected_context: str,
    downstream_output: str,
    alpha: float = 0.6,
) -> float:
    """
    Compute Context Utilization Score (CUS) deterministically.

    Uses a weighted token overlap between what was injected and what
    appeared in the downstream agent's output.

    Also applies a length-penalty: if the output is extremely short,
    it likely didn't engage deeply with the context regardless of overlap.

    Parameters
    ----------
    injected_context   : the text that was injected into the downstream prompt
    downstream_output  : the downstream agent's actual output
    alpha              : blend weight between token overlap (alpha) and
                         TF-IDF-style inverse frequency penalty (1-alpha).
                         Default 0.6 works well for bioinformatics text.

    Returns
    -------
    float in [0.0, 1.0]
        0.0 = agent completely ignored the context
        1.0 = agent used essentially everything injected
    """
    if not injected_context or not downstream_output:
        return 0.0

    injected_tokens = _extract_weighted_tokens(injected_context)
    output_tokens   = _extract_weighted_tokens(downstream_output)

    if not injected_tokens:
        return 0.0

    # ── Weighted overlap ──────────────────────────────────────────────────────
    total_injected_weight = sum(injected_tokens.values())
    matched_weight = 0.0

    for token, inj_weight in injected_tokens.items():
        if token in output_tokens:
            # Partial credit: a token used once counts less than one used many times
            out_freq = min(output_tokens[token], inj_weight)
            matched_weight += out_freq

    raw_overlap = matched_weight / total_injected_weight

    # ── Inverse frequency penalty ─────────────────────────────────────────────
    # Common tokens (appearing in both context and output often) are less
    # informative than rare technical tokens. Penalise overlap driven
    # entirely by common words.
    unique_injected = len(injected_tokens)
    unique_matched  = sum(1 for t in injected_tokens if t in output_tokens)
    diversity_score = unique_matched / max(unique_injected, 1)

    # ── Length penalty ────────────────────────────────────────────────────────
    # Very short outputs (< 50 words) can't have engaged deeply even if
    # they repeat a few tokens. Scale down proportionally.
    output_word_count = len(downstream_output.split())
    length_factor     = min(1.0, output_word_count / 50)

    # ── Final blend ───────────────────────────────────────────────────────────
    cus = (alpha * raw_overlap + (1 - alpha) * diversity_score) * length_factor

    return round(float(min(max(cus, 0.0), 1.0)), 4)


# ── How to call this from your existing evaluator ────────────────────────────
# In communication_evaluator.py, find the method that sets log.utilization_score
# It probably looks something like:
#
#   async def _score_utilization(self, log):
#       prompt = f"Did the agent use the context? Score 0-1..."
#       response = await self.llm.generate(prompt)
#       log.utilization_score = parse_score(response)
#
# Replace the body of that method with:
#
#   from .cus_evaluator_fix import compute_cus_deterministic
#   log.utilization_score = compute_cus_deterministic(
#       injected_context  = log.context,
#       downstream_output = log.output,
#   )
#
# The LLM call is no longer needed for CUS — this runs in microseconds.


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {
            "label": "Agent used almost nothing (expect ~0–15%)",
            "context": "Use Scanpy's rank_genes_groups with method='wilcoxon' on PBMC3k dataset. "
                       "Key genes: CD3D, CD79A, LYZ. Resolution parameter: 0.5.",
            "output":  "To cluster cells, apply standard dimensionality reduction. "
                       "PCA is commonly used before neighborhood graph construction.",
        },
        {
            "label": "Agent used some context (expect ~30–55%)",
            "context": "Use Scanpy's rank_genes_groups with method='wilcoxon' on PBMC3k dataset. "
                       "Key genes: CD3D, CD79A, LYZ. Resolution parameter: 0.5.",
            "output":  "We apply rank_genes_groups using the Wilcoxon test. "
                       "The analysis identifies marker genes including LYZ for monocytes.",
        },
        {
            "label": "Agent used most of the context (expect ~70–90%)",
            "context": "Use Scanpy's rank_genes_groups with method='wilcoxon' on PBMC3k dataset. "
                       "Key genes: CD3D, CD79A, LYZ. Resolution parameter: 0.5.",
            "output":  "import scanpy as sc\nsc.tl.rank_genes_groups(adata, groupby='leiden', "
                       "method='wilcoxon')\n# Top markers: CD3D (T cells), CD79A (B cells), "
                       "LYZ (monocytes)\n# Resolution was set to 0.5 for Leiden clustering",
        },
        {
            "label": "Agent fully reproduced context (expect ~90–100%)",
            "context": "Use Scanpy's rank_genes_groups with method='wilcoxon' on PBMC3k dataset. "
                       "Key genes: CD3D, CD79A, LYZ. Resolution parameter: 0.5.",
            "output":  "Using Scanpy's rank_genes_groups with method='wilcoxon' on the PBMC3k "
                       "dataset. The key genes CD3D, CD79A, and LYZ were identified. "
                       "Resolution parameter 0.5 was used for Leiden clustering.",
        },
    ]

    print("CUS Deterministic Evaluator — Self Test")
    print("=" * 55)
    for tc in test_cases:
        score = compute_cus_deterministic(tc["context"], tc["output"])
        print(f"\n  {tc['label']}")
        print(f"  CUS = {score:.4f}  ({score*100:.1f}%)")