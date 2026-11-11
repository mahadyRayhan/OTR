"""
context_layer.py — what crosses each edge of the agent DAG, and the policy that decides it.

Atoms. An upstream output is cut into atoms (atomize_output, the CUGCR atomizer: code blocks, named
entities, prose claims). typed_atoms() adds ids, the source node, and three refinements:

  CONSTRAINT  a prose claim that states a requirement or a hazard ("must", "at least 64 GB", "licence",
              "toxic"); protected: a policy should never drop it
  ARTIFACT    a code block too long to inline; the text is kept in the run's ArtifactStore and the
              atom carries a pointer ("[ARTIFACT a1: ...]")
  sensitive   a flag, not a type: the atom contains a nucleotide or protein sequence (>= 20 nt or
              >= 30 aa); a policy may keep it from external services

Policies. A ContextPolicy turns (task, receiver, upstream outputs) into the string the receiving agent
gets. Registered policies: "none" (task only), "raw" (task + full upstream text), "cugcr" (SciWeaver's
gated compression, engine._resolve_dependencies). Branches add their own with @register_policy.
route_context() applies a policy and records, in the run trace, every atom offered on the edge and
which ones reached the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from orchestrator import trace


# ── Entity stopwords for atomizer ─────────────────────────────────────────────
# NLTK English stopwords (title-cased) + domain-generic academic/bio terms that
# match the [A-Z][a-z]{4,} pattern but are not named entities.
def _build_entity_stopwords() -> frozenset:
    base: set[str] = set()
    try:
        import nltk
        from nltk.corpus import stopwords as _sw
        base = {w.title() for w in _sw.words("english")}
    except Exception:
        pass
    # Generic academic/bioinformatics prose words NLTK doesn't cover.
    # Expanded after observing false positives in telemetry (Local, Source, Solution…).
    extra = {
        # discourse / hedge words
        "Crucially", "Importantly", "Notably", "Particularly", "Essentially",
        "Specifically", "Generally", "Typically", "Commonly", "Frequently",
        "Paradoxically", "Conversely", "Similarly", "Consequently",
        # generic academic nouns
        "Refinement", "Improvement", "Detection", "Analysis", "Method",
        "Algorithm", "Approach", "Framework", "Process", "System",
        "Result", "Output", "Input", "Data", "Value", "Answer", "Paper",
        "Study", "Figure", "Table", "Section", "Example", "Step",
        "Performance", "Comparison", "Discussion", "Introduction",
        "Conclusion", "Abstract", "Source", "Solution", "Problem",
        "Challenge", "Phase", "Stage", "Version", "Component", "Module",
        "Structure", "Parameter", "Property", "Feature", "Factor", "Element",
        "Mechanism", "Strategy", "Technique", "Procedure", "Operation",
        # graph / math nouns (not tool names)
        "Communities", "Community", "Network", "Nodes", "Edges",
        "Graph", "Matrix", "Vector", "Score", "Partition", "Cluster",
        "Subgraph", "Connectivity", "Modularity", "Resolution",
        # generic adjectives that open sentences
        "Local", "Global", "Total", "Single", "Entire", "Original",
        "Novel", "Previous", "Current", "Recent", "Basic", "Advanced",
        "Standard", "Simple", "Complex", "Initial", "Final", "Primary",
        "Secondary", "Internal", "External", "Optimal", "Efficient",
        # verb-derived nouns / gerunds that start sentences
        "Moving", "Building", "Ensuring", "Computing", "Running",
        "Loading", "Saving", "Performing", "Generating", "Creating",
        "Based", "Using", "Given", "Through", "Across", "Along",
    }
    return frozenset(base | extra)

_ENTITY_STOPWORDS: frozenset = _build_entity_stopwords()

class AtomType(Enum):
    CODE_BLOCK = "code"
    NAMED_ENTITY = "entity"
    PROSE_CLAIM = "prose"
    # Produced only by typed_atoms(); atomize_output (CUGCR) never emits these.
    CONSTRAINT = "constraint"
    ARTIFACT = "artifact"

@dataclass
class ContextAtom:
    type: AtomType
    content: str
    is_protected: bool
    # Set by typed_atoms(); empty for plain CUGCR atoms.
    id: str = ""
    source: str = ""
    sensitive: bool = False


def atomize_output(raw_output: str) -> list[ContextAtom]:
    """Extracts protected code, biological entities, and prose deterministically."""
    atoms = []

    for match in re.finditer(r'```[\s\S]*?```', raw_output):
        atoms.append(ContextAtom(AtomType.CODE_BLOCK, match.group(), is_protected=True))

    prose_only = re.sub(r'```[\s\S]*?```', '', raw_output)
    # Strip markdown headers — words in "## Building Well-Connected Communities"
    # are structural section titles, not named entities. Without this, every
    # capitalized header word (Building, Source, Refined) becomes a false-positive entity.
    prose_only = re.sub(r'^#{1,6}\s+.*$', '', prose_only, flags=re.MULTILINE)
    # Strip bold/italic markers but keep the text (bold terms may be real entities)
    prose_only = re.sub(r'\*{1,3}', '', prose_only)

    bio_pattern = (
        r'\b('
        r'[A-Z]{2,}[0-9]*'              # all-caps acronyms: RNA, STAR, CRISPR
        r'|[A-Z][a-z]+[A-Z]\w*'         # CamelCase: Scanpy, AnnData, SAMtools
        r'|[A-Z]{2,}[a-z]+\w*'          # mixed start: DESeq2, HISAT2, BWA
        r'|[A-Z][a-z]+-\d+'             # hyphen-digit: Leiden-1
        r'|[A-Z][a-z]{4,}'              # long proper nouns ≥6 chars: Leiden, Louvain, Kallisto
        r')\b'
    )
    seen_entities: set[str] = set()
    for match in re.finditer(bio_pattern, prose_only):
        ent = match.group()
        if ent not in seen_entities and ent.title() not in _ENTITY_STOPWORDS:
            seen_entities.add(ent)
            atoms.append(ContextAtom(AtomType.NAMED_ENTITY, ent, is_protected=True))

    for sentence in prose_only.split('. '):
        if len(sentence.strip()) > 15:
            atoms.append(ContextAtom(AtomType.PROSE_CLAIM, sentence.strip() + ".", is_protected=False))

    return atoms


# ── Typed atoms ───────────────────────────────────────────────────────────────

ARTIFACT_CHARS = 2000  # code blocks longer than this travel by pointer
CONSTRAINT_CUES = re.compile(
    r"\b(?:must(?:\s+not)?|do\s+not|never|only\s+if|required|requires?\s+(?:at\s+least|a\s+minimum)|at\s+least|"
    r"no\s+more\s+than|minimum|maximum|limit(?:ed)?|licen[cs]e[ds]?|restricted|prohibited|caution|warning|"
    r"hazard\w*|toxic\w*|unsafe|patient\s+data|identifiable|consent|HIPAA|GDPR)\b",
    re.IGNORECASE)
SEQUENCE = re.compile(r"\b(?:[ACGTUN]{20,}|[ACDEFGHIKLMNPQRSTVWY]{30,})\b")


class ArtifactStore:
    """Large outputs of the current run, addressed by id; atoms carry pointers to them."""

    def __init__(self):
        self._items: dict[str, dict] = {}

    def put(self, text: str, source: str, kind: str) -> str:
        aid = f"a{len(self._items) + 1}"
        self._items[aid] = {"text": text, "source": source, "kind": kind}
        return aid

    def get(self, aid: str) -> str | None:
        item = self._items.get(aid)
        return item and item["text"]

    def pointer(self, aid: str) -> str:
        item = self._items[aid]
        return f"[ARTIFACT {aid}: {item['kind']} from node {item['source'][:8]}, {len(item['text'])} characters]"


def typed_atoms(text: str, source: str, store: ArtifactStore | None = None) -> list[ContextAtom]:
    """CUGCR atoms of one upstream output, with ids and the CONSTRAINT / ARTIFACT / sensitive refinements."""
    out = []
    for i, a in enumerate(atomize_output(text or "")):
        a.id, a.source = f"{source[:8]}#{i}", source
        a.sensitive = bool(SEQUENCE.search(a.content))
        if a.type == AtomType.PROSE_CLAIM and CONSTRAINT_CUES.search(a.content):
            a.type, a.is_protected = AtomType.CONSTRAINT, True
        elif a.type == AtomType.CODE_BLOCK and store is not None and len(a.content) > ARTIFACT_CHARS:
            aid = store.put(a.content, source, "code")
            a.type, a.content = AtomType.ARTIFACT, store.pointer(aid)
        out.append(a)
    return out


def atom_record(a: ContextAtom) -> dict:
    """The trace form of an atom (the full text, so a replay can rebuild any subset)."""
    return {"id": a.id, "source": a.source, "type": a.type.value, "protected": a.is_protected,
            "sensitive": a.sensitive, "text": a.content}


# ── Policies ──────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    """Everything a policy may use to build one node's input."""
    task: str                    # the node's own task text
    receiver: str                # agent name of the receiving node
    upstream: dict[str, str]     # parent node id -> its output
    user_query: str = ""
    atoms: list[ContextAtom] = field(default_factory=list)  # typed atoms of all upstream outputs


class ContextPolicy:
    name = "base"

    async def route(self, engine, edge: Edge) -> str:
        raise NotImplementedError


_POLICIES: dict[str, ContextPolicy] = {}


def register_policy(cls):
    _POLICIES[cls.name] = cls()
    return cls


def get_policy(name: str) -> ContextPolicy:
    if name not in _POLICIES:
        raise KeyError(f"unknown context policy {name!r}; registered: {sorted(_POLICIES)}")
    return _POLICIES[name]


def policy_for_mode(ablation_mode: str) -> str:
    """The engine's ablation modes map onto policies; anything else uses SciWeaver's CUGCR."""
    return {"none": "none", "raw": "raw"}.get(ablation_mode, "cugcr")


@register_policy
class NoContext(ContextPolicy):
    name = "none"

    async def route(self, engine, edge):
        return edge.task


@register_policy
class RawContext(ContextPolicy):
    name = "raw"

    async def route(self, engine, edge):
        context = "\n".join(f"--- Result ---\n{v}" for v in edge.upstream.values())
        return f"{edge.task}\n\n[UPSTREAM CONTEXT]:\n{context}"


@register_policy
class Cugcr(ContextPolicy):
    """SciWeaver's gated compression, unchanged: raw if within the receiver's budget, else CUGCR."""
    name = "cugcr"

    async def route(self, engine, edge):
        return await engine._resolve_dependencies(edge.task, list(edge.upstream), edge.upstream, "adaptive",
                                                  downstream_agent=edge.receiver)


async def route_context(engine, policy_name: str, task: str, receiver: str, upstream: dict[str, str],
                        user_query: str = "", store: ArtifactStore | None = None) -> str:
    """Build the receiver's input with a policy, and record the edge (offered and kept atoms) in the trace."""
    atoms = [a for nid, out in upstream.items() for a in typed_atoms(out, nid, store)]
    edge = Edge(task=task, receiver=receiver, upstream=upstream, user_query=user_query, atoms=atoms)
    text = await get_policy(policy_name).route(engine, edge)
    # An atom reached the agent if its text appears in the input; an artifact, if its pointer or its full text does.
    def reached(a: ContextAtom) -> bool:
        forms = [a.content.rstrip(".").strip()]
        if a.type == AtomType.ARTIFACT and store is not None:
            forms.append((store.get(a.content.split()[1].rstrip(":")) or "").strip())
        return any(f and f in text for f in forms)
    kept = [a.id for a in atoms if reached(a)]
    trace.record_edge([atom_record(a) for a in atoms], kept, policy_name, text,
                      upstream_chars=sum(len(v) for v in upstream.values()), input_chars=len(text))
    return text
