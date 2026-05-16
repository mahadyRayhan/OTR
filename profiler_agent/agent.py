"""
user_profiler.py
═══════════════════════════════════════════════════════════════════════════════
Dynamic User Intent Profiler for Bioinformatics AI Agents

Core idea (your original):
    Maintain a vector of "response-style weights" that evolves over time as
    the user interacts, so the system can tailor answer style to the user's
    actual preference — not a static template.

Literature grounding
────────────────────
Dimension design:
    Anderson & Krathwohl (2001). A Taxonomy for Learning, Teaching, and
        Assessing (Bloom's Revised Taxonomy). Longman.
        → Three dimensions are collapsed from six cognitive levels:
          CONCEPTUAL  = Remember + Understand
          PROCEDURAL  = Apply + Create
          ANALYTICAL  = Analyze + Evaluate

    Broder (2002). A Taxonomy of Web Search. SIGIR Forum.
        → Query intent has structure (Informational / Navigational /
          Transactional) — confirms that intent classification is tractable.

Bayesian update (replaces EMA):
    Johnson (1960). The Dirichlet Distribution.
    Spiegelhalter & Lauritzen (1990). Sequential Updating of Conditional
        Probabilities on Directed Graphical Structures. Networks, 20(5).
        → Dirichlet(α₁,…,αₖ) is the conjugate prior for proportions.
          Updates are additive: αᵢ ← αᵢ + observed_i.
          No learning rate needed; confidence grows naturally with n.

User modeling / adaptive systems:
    Kobsa (2001). Generic User Modelling Systems. UMUAI, 11(1).
        → Canonical reference for adaptive profiles in intelligent systems.
    Kay (2000). Stereotypes, Student Models and Scrutability. ITS 2000.
        → Distinguishes short-term (session) from long-term (trait) models.

Context-switch detection:
    Kullback & Leibler (1951). On Information and Sufficiency. AoMS, 22(1).
        → KL divergence D_KL(q ∥ profile) quantifies "surprise" — how
          different the current query is from the settled profile.
    Adams (1999). Detecting changes in the onset of intent sequences.
          → Change-point framing for sudden user mindset switches.

Why Dirichlet beats EMA here
─────────────────────────────
    EMA:       α_new = α * (1-lr) + q * lr           (fixed forgetting rate)
    Dirichlet: α_vec ← α_vec + q * weight            (additive pseudo-counts)

    Dirichlet advantage 1 — no magic learning rate:
        EMA lr=0.6 means ONE query shifts the profile by 60%.
        Dirichlet naturally anneals: early queries matter more, later queries
        incrementally refine — because sum(α) grows with each update.

    Dirichlet advantage 2 — free uncertainty tracking:
        Var[Xᵢ] = αᵢ(S-αᵢ) / (S²(S+1))  where S = Σαⱼ
        Low variance → settled profile; high variance → still learning.

    Dirichlet advantage 3 — principled context-switch response:
        On switch detection (high KL), inject extra weight W_surge into the
        update. This is equivalent to observing W_surge "surprise" queries
        simultaneously, pulling the Dirichlet mean toward the new intent fast,
        then naturally annealing again — no separate tunable parameter.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DIMENSION DEFINITIONS
# Grounded in Bloom's Revised Taxonomy (Anderson & Krathwohl 2001).
# Each dimension maps to a cluster of cognitive levels and query types.
# ──────────────────────────────────────────────────────────────────────────────

class Dim(str, Enum):
    """
    Three response-style dimensions derived from Bloom's Revised Taxonomy.

    CONCEPTUAL  ← Bloom Levels 1-2: Remember, Understand
        Queries: "What is X?", "How does X work?", "Explain X"
        Answer style: definitions, mechanisms, theory, analogies

    PROCEDURAL  ← Bloom Levels 3-6 (implementation axis): Apply, Create
        Queries: "Write a script to…", "Implement X", "Debug this"
        Answer style: runnable code, step-by-step instructions, tools

    ANALYTICAL  ← Bloom Levels 4-5: Analyze, Evaluate
        Queries: "Compare X vs Y", "What are the trade-offs?",
                 "Critique this approach", "Is X better than Y?"
        Answer style: comparisons, literature citations, pros/cons,
                      quantitative evidence, expert opinions
    """
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    ANALYTICAL = "analytical"


DIMS: List[Dim] = list(Dim)
N_DIM: int = len(DIMS)


# ──────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC INTENT CLASSIFIER (no LLM required)
# Used as primary path; LLM refines when available.
# ──────────────────────────────────────────────────────────────────────────────

# Each rule: (compiled regex, {Dim: score})  — scores need not sum to 100;
# they are normalised after accumulation across all matching rules.
_INTENT_RULES: List[Tuple[re.Pattern, Dict[Dim, float]]] = [
    # ── Strong PROCEDURAL signals ──────────────────────────────────────────
    (re.compile(
        r"\b(write|implement|code|script|debug|fix|program|build|"
        r"function|class|snakemake|nextflow|biopython|pandas|numpy|"
        r"deploy|aws|docker|kubernetes|pipeline code)\b", re.I),
     {Dim.PROCEDURAL: 60, Dim.CONCEPTUAL: 20, Dim.ANALYTICAL: 20}),

    # ── Strong ANALYTICAL signals ──────────────────────────────────────────
    (re.compile(
        r"\b(compare|versus|vs\.?|trade.?off|pros and cons|difference|"
        r"better|worse|evaluate|critique|assess|benchmark|review|"
        r"literature|state of the art|which is|advantage|disadvantage)\b", re.I),
     {Dim.ANALYTICAL: 60, Dim.CONCEPTUAL: 25, Dim.PROCEDURAL: 15}),

    # ── Strong CONCEPTUAL signals ──────────────────────────────────────────
    (re.compile(
        r"\b(what is|what are|how does|explain|define|describe|overview|"
        r"concept|theory|mechanism|principle|background|history|"
        r"understand|intuition|idea behind)\b", re.I),
     {Dim.CONCEPTUAL: 60, Dim.ANALYTICAL: 25, Dim.PROCEDURAL: 15}),

    # ── Moderate ANALYTICAL signals ────────────────────────────────────────
    (re.compile(
        r"\b(why|should I|is it true|verify|confirm|deep dive|"
        r"in.?depth|thorough|rigorous|expert|paper|publication|"
        r"citation|research|study|journal)\b", re.I),
     {Dim.ANALYTICAL: 45, Dim.CONCEPTUAL: 35, Dim.PROCEDURAL: 20}),

    # ── Moderate PROCEDURAL signals ────────────────────────────────────────
    (re.compile(
        r"\b(how to|step by step|tutorial|guide|example|run|execute|"
        r"install|setup|configure|use|apply)\b", re.I),
     {Dim.PROCEDURAL: 45, Dim.CONCEPTUAL: 35, Dim.ANALYTICAL: 20}),
]

# Uniform fallback when no rule fires
_UNIFORM: Dict[Dim, float] = {d: 100.0 / N_DIM for d in DIMS}


def _classify_deterministic(query: str) -> Dict[Dim, float]:
    """
    Deterministically classify query intent into dimension weights.
    Applies all matching rules and averages their score vectors.
    Returns a normalised dict summing to 100.0.
    """
    accumulated: Dict[Dim, float] = {d: 0.0 for d in DIMS}
    n_matched = 0

    for pattern, scores in _INTENT_RULES:
        if pattern.search(query):
            for dim in DIMS:
                accumulated[dim] += scores.get(dim, 0.0)
            n_matched += 1

    if n_matched == 0:
        return dict(_UNIFORM)

    # Average across matched rules
    averaged = {d: accumulated[d] / n_matched for d in DIMS}

    # Normalise to 100
    total = sum(averaged.values())
    return {d: (averaged[d] / total) * 100.0 for d in DIMS}


# ──────────────────────────────────────────────────────────────────────────────
# BAYESIAN UPDATE MATH (Dirichlet conjugate prior)
# ──────────────────────────────────────────────────────────────────────────────

def _kl_divergence(p: Dict[Dim, float], q: Dict[Dim, float]) -> float:
    """
    Compute KL divergence D_KL(p ∥ q) in nats.

    Reference:
        Kullback & Leibler (1951). On Information and Sufficiency.
        Annals of Mathematical Statistics, 22(1), 79-86.

    Small values (< 0.1) → current query is consistent with profile.
    Large values (> 0.5) → significant context switch.

    p and q are treated as normalised probability distributions (÷100).
    A small epsilon guards against log(0).
    """
    eps = 1e-9
    return sum(
        (p[d] / 100.0) * math.log((p[d] / 100.0 + eps) / (q[d] / 100.0 + eps))
        for d in DIMS
    )


@dataclass
class UserProfile:
    """
    Bayesian user profile backed by a Dirichlet distribution.

    Attributes
    ──────────
    alpha       Dirichlet concentration parameters. Start at alpha_init for
                each dimension (uniform prior). Each query adds to these.
                profile[i] = alpha[i] / sum(alpha).

    alpha_init  Strength of the prior. Higher = more queries needed to shift
                the profile. Default 10 ≈ "seen 10 balanced queries already",
                i.e. robust to the first 1-2 queries being extreme outliers.
                Ref: Kobsa (2001) recommends weak priors for fast adaptation.

    session_alpha   Separate Dirichlet for the current session only (resets
                    each session). Reflects Kay (2000) short-term model.

    switch_history  Log of detected context switches for introspection.
    """

    alpha_init: float = 10.0
    kl_switch_threshold: float = 0.4   # nats; above this → context switch
    surge_weight: float = 8.0          # extra pseudo-count mass on switch

    # Long-term Dirichlet parameters (persist across sessions)
    alpha: Dict[Dim, float] = field(default_factory=dict)
    # Short-term / session Dirichlet parameters (reset per session)
    session_alpha: Dict[Dim, float] = field(default_factory=dict)

    # Metadata
    n_queries: int = 0
    switch_history: List[Tuple[int, float, Dict[Dim, float]]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        for container in (self.alpha, self.session_alpha):
            if not container:
                for d in DIMS:
                    container[d] = self.alpha_init

    # ── Core properties ───────────────────────────────────────────────────────

    @property
    def profile(self) -> Dict[Dim, float]:
        """
        Current long-term profile as percentages summing to 100.
        = Dirichlet mean: αᵢ / Σαⱼ
        """
        total = sum(self.alpha.values())
        return {d: round((self.alpha[d] / total) * 100.0, 1) for d in DIMS}

    @property
    def session_profile(self) -> Dict[Dim, float]:
        """Current session profile as percentages."""
        total = sum(self.session_alpha.values())
        return {d: round((self.session_alpha[d] / total) * 100.0, 1) for d in DIMS}

    @property
    def confidence(self) -> Dict[Dim, float]:
        """
        Certainty in each dimension as 1 - coefficient_of_variation.
        Derived from Dirichlet variance:
            Var[Xᵢ] = αᵢ(S - αᵢ) / (S²(S + 1))    where S = Σαⱼ

        Returns a value in [0, 1] per dimension:
            0.0 = completely uncertain (early queries, flat profile)
            1.0 = fully certain (profile has strongly settled)

        Practical interpretation:
            < 0.5  : profile still forming — treat weights loosely
            0.5-0.8: moderate confidence — profile is useful
            > 0.8  : high confidence — profile is well-established
        """
        S = sum(self.alpha.values())
        confidence = {}
        for d in DIMS:
            ai = self.alpha[d]
            var = (ai * (S - ai)) / (S ** 2 * (S + 1)) if S > 1 else 1.0
            # Normalise: max variance of Dirichlet(1,1,...,1) = (1/k)(1-1/k)/(k+1)
            max_var = (1.0 / N_DIM) * (1.0 - 1.0 / N_DIM) / (N_DIM + 1)
            confidence[d] = round(max(0.0, 1.0 - var / max_var), 3)
        return confidence

    @property
    def overall_confidence(self) -> float:
        """Scalar confidence in [0,1]: harmonic mean across dimensions."""
        vals = list(self.confidence.values())
        if any(v == 0 for v in vals):
            return 0.0
        return round(N_DIM / sum(1.0 / v for v in vals), 3)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, intent: Dict[Dim, float]) -> bool:
        """
        Bayesian Dirichlet update given an observed intent vector.

        The intent dict (values summing to 100) represents the "evidence"
        from one query. We add it as pseudo-counts to alpha, scaled by a
        weight of 1.0 (standard) or surge_weight (on context switch).

        Returns True if a context switch was detected.

        Math:
            α_new = α_old + intent_normalised * weight
            profile_new = α_new / sum(α_new)

        Because sum(α) grows monotonically, each subsequent query has
        *proportionally* less effect — natural Bayesian annealing.
        No learning rate needed.
        """
        self.n_queries += 1
        current_profile = self.profile

        # Detect context switch via KL divergence
        kl = _kl_divergence(intent, current_profile)
        switched = kl > self.kl_switch_threshold
        weight = self.surge_weight if switched else 1.0

        if switched:
            logger.info(
                "[UserProfiler] Context switch detected at query %d "
                "(KL=%.3f > threshold=%.2f). Applying surge weight %.1f.",
                self.n_queries, kl, self.kl_switch_threshold, weight,
            )
            self.switch_history.append((self.n_queries, round(kl, 3), dict(intent)))

        # Normalise intent to [0, 1] range, then add weighted pseudo-counts
        intent_norm = {d: intent[d] / 100.0 for d in DIMS}
        for d in DIMS:
            self.alpha[d] += intent_norm[d] * weight
            self.session_alpha[d] += intent_norm[d] * weight

        return switched

    def reset_session(self) -> None:
        """Reset the short-term session model while preserving long-term profile."""
        self.session_alpha = {d: self.alpha_init for d in DIMS}
        logger.debug("[UserProfiler] Session model reset.")

    def reset_all(self) -> None:
        """Full reset — useful for benchmarking or new users."""
        self.alpha = {d: self.alpha_init for d in DIMS}
        self.session_alpha = {d: self.alpha_init for d in DIMS}
        self.n_queries = 0
        self.switch_history.clear()
        logger.debug("[UserProfiler] Full profile reset.")

    # ── Style guideline ───────────────────────────────────────────────────────

    def get_style_guideline(self, *, use_session: bool = False) -> str:
        """
        Translate the mathematical profile into a prompt constraint string
        for the downstream synthesiser agent.

        Args:
            use_session: If True, uses the session-only profile (more reactive).
                         If False, uses the long-term profile (more stable).
        """
        prof = self.session_profile if use_session else self.profile
        conf = self.overall_confidence
        conf_label = (
            "still forming — treat these as soft hints"
            if conf < 0.5
            else "moderate confidence" if conf < 0.8
            else "high confidence — follow these proportions strictly"
        )

        lines = [
            "RESPONSE STYLE PROFILE (user-adaptive):",
            f"  Conceptual / Theory:         {prof[Dim.CONCEPTUAL]:5.1f}%",
            f"  Procedural / Code:           {prof[Dim.PROCEDURAL]:5.1f}%",
            f"  Analytical / Evaluation:     {prof[Dim.ANALYTICAL]:5.1f}%",
            f"  Profile confidence: {conf:.2f} ({conf_label})",
            "",
            "INSTRUCTIONS:",
        ]

        dominant = max(prof, key=prof.get)
        if dominant == Dim.PROCEDURAL and prof[Dim.PROCEDURAL] > 45:
            lines.append("  - Lead with working code; keep explanations brief.")
        elif dominant == Dim.ANALYTICAL and prof[Dim.ANALYTICAL] > 45:
            lines.append("  - Lead with comparisons, evidence, and trade-offs.")
            lines.append("  - Cite relevant literature or benchmarks where possible.")
        elif dominant == Dim.CONCEPTUAL and prof[Dim.CONCEPTUAL] > 45:
            lines.append("  - Lead with clear conceptual explanation; use analogies.")
            lines.append("  - Avoid code unless it directly illustrates the concept.")
        else:
            lines.append("  - Balance explanation, code, and analysis proportionally.")

        if len(self.switch_history) > 0:
            last_switch_q, last_kl, _ = self.switch_history[-1]
            if self.n_queries - last_switch_q <= 2:
                lines.append(
                    f"  - NOTE: User recently switched intent (query {last_switch_q}, "
                    f"KL={last_kl:.2f}). Prioritise the current query's explicit style."
                )
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable state dump for logging / debugging."""
        p = self.profile
        c = self.confidence
        return (
            f"UserProfile(n={self.n_queries}, "
            f"conceptual={p[Dim.CONCEPTUAL]:.1f}% [conf={c[Dim.CONCEPTUAL]:.2f}], "
            f"procedural={p[Dim.PROCEDURAL]:.1f}% [conf={c[Dim.PROCEDURAL]:.2f}], "
            f"analytical={p[Dim.ANALYTICAL]:.1f}% [conf={c[Dim.ANALYTICAL]:.2f}], "
            f"switches={len(self.switch_history)})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC PROFILER AGENT
# Wraps UserProfile with async LLM refinement and deterministic fallback.
# ──────────────────────────────────────────────────────────────────────────────

# 10s caused frequent timeouts on first API call (cold start + model latency).
_LLM_TIMEOUT_S: float = 30.0

_LLM_CLASSIFICATION_PROMPT = """
Analyze this bioinformatics query and determine what kind of response the user wants.
Distribute EXACTLY 100 points across three dimensions:

  "conceptual"  — definitions, theory, mechanisms, conceptual understanding
  "procedural"  — code, scripts, implementation, step-by-step instructions
  "analytical"  — comparisons, trade-offs, literature, deep evaluation

Return ONLY valid JSON — no prose, no markdown, no explanation:
{{"conceptual": <int>, "procedural": <int>, "analytical": <int>}}

User Query: "{query}"
""".strip()


class DynamicProfiler:
    """
    Dynamic User Intent Profiler.

    Public API
    ──────────
    update_profile(query, model)  → dict   async; updates profile, returns percentages
    get_style_guideline()         → str    prompt constraint for synthesiser
    profile.summary()             → str    debug/logging snapshot
    profile.reset_session()              reset session model (keep long-term)
    profile.reset_all()                  full reset

    Architecture
    ────────────
    1. Deterministic classifier  (_classify_deterministic) — always runs, O(k)
    2. LLM refiner               (_llm_classify)           — runs if model provided
    3. Blending                  — LLM result weighted 70%, deterministic 30%
       (deterministic keeps LLM hallucinations in check)
    4. Bayesian update           (profile.update)           — Dirichlet conjugate prior
    5. Context switch detection  (_kl_divergence)           — embedded in update()
    """

    def __init__(
        self,
        *,
        alpha_init: float = 10.0,
        kl_switch_threshold: float = 0.4,
        surge_weight: float = 8.0,
        llm_blend_weight: float = 0.7,
    ) -> None:
        """
        Args:
            alpha_init: Dirichlet prior strength. Higher = slower to shift.
                        10 ≈ "equivalent to having seen 10 balanced queries".
                        Recommended range: 5–20.
            kl_switch_threshold: KL divergence above which a context switch is
                        declared and surge_weight is applied. 0.4 nats ≈ 33%
                        relative change in dominant dimension. Range: 0.2–0.8.
            surge_weight: Extra pseudo-count mass injected on context switch.
                        8.0 means the switch query counts as 8 normal queries.
                        This gives fast adaptation while still preserving prior.
            llm_blend_weight: Fraction of the LLM's classification that overrides
                        the deterministic baseline. 0.7 means LLM drives 70%.
        """
        self.profile = UserProfile(
            alpha_init=alpha_init,
            kl_switch_threshold=kl_switch_threshold,
            surge_weight=surge_weight,
        )
        self._llm_blend_weight = llm_blend_weight

    # ── LLM classification ────────────────────────────────────────────────────

    async def _llm_classify(
        self, query: str, model: genai.GenerativeModel
    ) -> Optional[Dict[Dim, float]]:
        """
        Ask the LLM to classify query intent.
        Returns None on failure, timeout, or malformed output.
        Timeout-guarded (asyncio.wait_for) to prevent indefinite blocking.
        """
        prompt = _LLM_CLASSIFICATION_PROMPT.format(query=query)
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(model.generate_content, prompt),
                timeout=_LLM_TIMEOUT_S,
            )
            text = res.text.replace("```json", "").replace("```", "").strip()
            raw: Dict[str, float] = json.loads(text)

            # Validate keys
            result: Dict[Dim, float] = {}
            for dim in DIMS:
                val = raw.get(dim.value, None)
                if val is None:
                    raise KeyError(f"Missing key '{dim.value}' in LLM response")
                result[dim] = float(val)

            # Normalise
            total = sum(result.values())
            if total <= 0:
                raise ValueError("LLM returned all-zero intent vector.")
            return {d: (result[d] / total) * 100.0 for d in DIMS}

        except asyncio.TimeoutError:
            logger.warning(
                "[UserProfiler] LLM classification timed out after %.1fs. "
                "Using deterministic classifier only.",
                _LLM_TIMEOUT_S,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "[UserProfiler] LLM returned malformed intent (%s). "
                "Using deterministic classifier only.",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "[UserProfiler] LLM classification failed (%s). "
                "Using deterministic classifier only.",
                exc,
            )
        return None

    # ── Main update ───────────────────────────────────────────────────────────

    async def update_profile(
        self,
        query: str,
        model: Optional[genai.GenerativeModel] = None,
    ) -> Dict[str, float]:
        """
        Classify the query's intent and update the Bayesian user profile.

        Pipeline:
            1. Always run deterministic classifier (fast, reliable).
            2. If model provided, run LLM classifier (richer, slower).
            3. Blend: intent = det * (1-w) + llm * w   (w = llm_blend_weight)
            4. Dirichlet update on profile (with optional context-switch surge).

        Returns:
            Dict with string keys (dim.value → percentage float), summing to 100.
        """
        # Step 1: Deterministic baseline
        det_intent = _classify_deterministic(query)
        logger.debug("[UserProfiler] Deterministic intent: %s", det_intent)

        # Step 2: LLM refinement
        final_intent: Dict[Dim, float]
        if model is not None:
            llm_intent = await self._llm_classify(query, model)
            if llm_intent is not None:
                logger.debug("[UserProfiler] LLM intent: %s", llm_intent)
                w = self._llm_blend_weight
                blended = {d: det_intent[d] * (1 - w) + llm_intent[d] * w for d in DIMS}
                total = sum(blended.values())
                final_intent = {d: (blended[d] / total) * 100.0 for d in DIMS}
            else:
                final_intent = det_intent
        else:
            final_intent = det_intent

        # Step 3: Bayesian update
        switched = self.profile.update(final_intent)
        if switched:
            logger.info("[UserProfiler] After switch: %s", self.profile.summary())
        else:
            logger.debug("[UserProfiler] Updated: %s", self.profile.summary())

        # Return with string keys for JSON compatibility
        return {d.value: v for d, v in self.profile.profile.items()}

    def get_style_guideline(self, *, use_session: bool = False) -> str:
        """Prompt constraint string for the synthesiser agent."""
        return self.profile.get_style_guideline(use_session=use_session)

    def reset_session(self) -> None:
        self.profile.reset_session()

    def reset_all(self) -> None:
        self.profile.reset_all()


# ──────────────────────────────────────────────────────────────────────────────
# STATELESS SINGLE-QUERY CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────

def classify_query(query: str) -> Dict[str, float]:
    """
    Classify a single query's intent without touching the global profile state.
    Returns {dim.value: percentage} summing to 100, using the deterministic path.
    Use this in benchmark mode or anywhere you want a clean per-query signal.
    """
    return {d.value: v for d, v in _classify_deterministic(query).items()}


def dominant_dim(query: str) -> str:
    """Return the single dominant dimension name ('conceptual'|'procedural'|'analytical')."""
    scores = _classify_deterministic(query)
    return max(scores, key=scores.get).value


# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL SINGLETON
# ──────────────────────────────────────────────────────────────────────────────

# Shared profiler that persists across the session.
# Call reset_all() for a new user; reset_session() for a new conversation.
user_profiler = DynamicProfiler()


# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION DEMO (no model required — runs deterministic path only)
# ──────────────────────────────────────────────────────────────────────────────

async def demo_simulation() -> None:
    """
    Simulate a realistic user session without a live LLM.
    Shows how the profile evolves and how context switches are detected.
    """
    logging.basicConfig(level=logging.INFO)

    profiler = DynamicProfiler(
        alpha_init=3.0,       # weaker prior → faster learning in demo
        kl_switch_threshold=0.2,  # lower threshold for clearer demo
        surge_weight=8.0,
    )

    # Simulated user session: starts analysis-heavy, switches to code
    session: List[Tuple[str, str]] = [
        ("q1", "Explain the difference between BWA and Bowtie2 for short-read alignment"),
        ("q2", "Compare the accuracy and speed trade-offs of BWA-MEM vs Bowtie2 in detail"),
        ("q3", "What does the literature say about the best aligner for RNA-seq data?"),
        ("q4", "Evaluate STAR vs HISAT2 for splice-aware alignment — pros, cons, benchmarks"),
        # ── Sudden context switch to code ──────────────────────────────────
        ("q5", "Write a Python script that runs BWA-MEM on a FASTQ file"),
        ("q6", "Debug this Snakemake rule — it crashes with a missing output error"),
        ("q7", "Implement a pipeline in Nextflow for paired-end RNA-seq"),
    ]

    sep = "─" * 70
    print(sep)
    print("Dynamic User Profiler — Session Simulation (deterministic path)")
    print(sep)
    print(f"{'Query':<6}  {'Conceptual':>11}  {'Procedural':>11}  {'Analytical':>11}  "
          f"{'Conf':>6}  {'Switch':>7}")
    print(sep)

    for qid, query in session:
        await profiler.update_profile(query, model=None)  # no LLM
        p = profiler.profile.profile
        conf = profiler.profile.overall_confidence
        switched = (
            any(sw[0] == profiler.profile.n_queries for sw in profiler.profile.switch_history)
        )
        flag = "  *** SWITCH" if switched else ""
        print(
            f"{qid:<6}  "
            f"{p[Dim.CONCEPTUAL]:>10.1f}%  "
            f"{p[Dim.PROCEDURAL]:>10.1f}%  "
            f"{p[Dim.ANALYTICAL]:>10.1f}%  "
            f"{conf:>6.3f}  "
            f"{flag}"
        )
        print(f"       query: {query[:65]}{'…' if len(query)>65 else ''}")

    print(sep)
    print("\nFinal style guideline:")
    print(profiler.get_style_guideline())
    print(sep)
    print("\nSwitch history:")
    for q_n, kl, intent in profiler.profile.switch_history:
        print(f"  Query {q_n}: KL={kl:.3f} | intent at switch: {intent}")


if __name__ == "__main__":
    asyncio.run(demo_simulation())