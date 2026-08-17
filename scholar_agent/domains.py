"""
domains.py — Domain registry for the scholar agent
===================================================
Single source of truth for which research domains the scholar agent serves,
which Semantic Scholar topics seed each domain's expert cache, and how each
domain should be described back to the LLM.

Adding a fourth domain = adding one entry to DOMAIN_TOPICS + DOMAIN_LABELS,
then running:

    python scholar_agent/build_scholar_cache.py --domain <name>

This module is deliberately dependency-free (stdlib only) so both the cache
builder and the runtime agent can import it without pulling in chromadb.
"""

from __future__ import annotations

# ── Canonical domain keys ────────────────────────────────────────────────────
BIOINFORMATICS    = "bioinformatics"
MATERIALS_SCIENCE = "materials_science"
HEALTH_SCIENCE    = "health_science"

DEFAULT_DOMAIN = BIOINFORMATICS

ALL_DOMAINS = (BIOINFORMATICS, MATERIALS_SCIENCE, HEALTH_SCIENCE)


# ── Human-readable labels ────────────────────────────────────────────────────
# Used as the expertise-area fallback when an author has no field-of-study
# tags, and injected into the topic-extraction prompt.
DOMAIN_LABELS: dict[str, str] = {
    BIOINFORMATICS:    "bioinformatics",
    MATERIALS_SCIENCE: "materials science",
    HEALTH_SCIENCE:    "health science",
}


# ── Topic-extraction hints (few-shot signal for extract_search_topic) ────────
# The original prompt hard-coded bioinformatics examples, which biased topic
# extraction for any non-bio query. Each domain now supplies its own KEEP
# examples and rewrite pairs.
DOMAIN_SEARCH_HINTS: dict[str, dict[str, str]] = {
    BIOINFORMATICS: {
        "keep": (
            "  • Specific tool or software names        (e.g. 'Kallisto', 'STAR aligner')\n"
            "  • Specific biological subtypes            (e.g. 'base editing', not just 'CRISPR';\n"
            "                                              'adenine base editor', not just 'gene editor')\n"
            "  • Domain-narrowing modifiers              (e.g. 'off-target effects', 'pseudoalignment')"
        ),
        "examples": (
            "      'retrieve parameter values from the Kallisto paper'  → 'Kallisto RNA-seq pseudoalignment'\n"
            "      'extract exact values from BWA documentation'        → 'BWA short read alignment'\n"
            "      'get detection methods from the CRISPR article'      → 'CRISPR base editing'"
        ),
    },
    MATERIALS_SCIENCE: {
        "keep": (
            "  • Specific material systems               (e.g. 'perovskite solar cell', not just 'solar cell';\n"
            "                                              'high entropy alloy', not just 'alloy')\n"
            "  • Simulation codes and instruments        (e.g. 'VASP', 'LAMMPS', 'Quantum ESPRESSO', 'XRD')\n"
            "  • Property or process modifiers           (e.g. 'thermal conductivity', 'grain boundary',\n"
            "                                              'thin film deposition', 'cycling stability')"
        ),
        "examples": (
            "      'retrieve lattice parameters from the VASP paper'    → 'VASP density functional theory'\n"
            "      'extract synthesis conditions from the CNT review'   → 'carbon nanotube synthesis'\n"
            "      'get the capacity numbers from the cathode study'    → 'lithium ion battery cathode'"
        ),
    },
    HEALTH_SCIENCE: {
        "keep": (
            "  • Specific conditions and populations     (e.g. 'type 2 diabetes', not just 'disease';\n"
            "                                              'preterm infants', not just 'patients')\n"
            "  • Study designs and methodologies         (e.g. 'randomized controlled trial',\n"
            "                                              'prospective cohort', 'meta-analysis')\n"
            "  • Outcome and intervention modifiers      (e.g. 'medication adherence', 'readmission rate',\n"
            "                                              'cost effectiveness', 'screening uptake')"
        ),
        "examples": (
            "      'retrieve the effect sizes from the RCT paper'       → 'randomized controlled trial outcomes'\n"
            "      'extract risk ratios from the cohort study'          → 'prospective cohort risk factors'\n"
            "      'get the model AUC from the sepsis prediction work'  → 'sepsis clinical prediction model'"
        ),
    },
}


# ── Seed topics per domain ───────────────────────────────────────────────────
# Each entry is (semantic_scholar_query, tag_string). Tags are free-form
# provenance labels recorded in search_topics.query_tags — for bioinformatics
# they map to the existing BQ/SQ benchmark IDs; new domains use slugs until a
# matching benchmark set exists.
DOMAIN_TOPICS: dict[str, list[tuple[str, str]]] = {

    # ─────────────────────────────────────────────────────────────────────────
    BIOINFORMATICS: [
        # Benchmark G1: Full-Stack Pipeline
        ("Leiden community detection single-cell genomics",         "BQ01,BQ06,SQ-leiden"),
        ("STAR RNA-seq splice junction alignment",                  "BQ02,BQ08,BQ27"),
        ("DESeq2 differential expression RNA-seq",                  "BQ03,BQ10,BQ26"),
        ("SAM BAM pysam sequencing alignment format",               "BQ04,BQ29"),
        ("Seurat weighted nearest neighbor multimodal single-cell", "BQ05,BQ09"),
        # Benchmark G2: Coreference / Knowledge Transfer
        ("CRISPR base editing cytosine adenine",                    "BQ07,BQ30,SQ09"),
        # Benchmark G4: Missing Data
        ("Monocle3 pseudotime trajectory inference single-cell",    "BQ16"),
        ("Salmon Kallisto RNA-seq pseudoalignment quantification",  "BQ17"),
        ("MAGIC imputation single-cell gene expression",            "BQ18"),
        ("GATK HaplotypeCaller variant calling whole genome",       "BQ20"),
        # Eval queries (SQ01-SQ10)
        ("single-cell RNA sequencing trajectory analysis",          "SQ01"),
        ("spatial transcriptomics gene expression imaging",         "SQ02"),
        ("chromatin accessibility ATAC-seq epigenomics",            "SQ03"),
        ("multi-omics data integration single-cell",                "SQ04"),
        ("proteomics transcriptomics integration mass spectrometry","SQ05"),
        ("deep learning neural network genomics DNA sequence",      "SQ06"),
        ("graph neural network protein structure prediction",       "SQ07"),
        ("cancer single-cell sequencing tumor microenvironment",    "SQ08"),
        ("Oxford Nanopore long-read sequencing bioinformatics",     "SQ10"),
        # Extra broad coverage
        ("Scanpy single-cell analysis Python",                      "scanpy"),
        ("scRNA-seq cell clustering dimensionality reduction UMAP", "scrna-cluster"),
        ("metagenomics microbiome 16S rRNA sequencing",             "metagenomics"),
        ("AlphaFold protein structure deep learning",               "alphafold"),
        ("CRISPR screen functional genomics genome-wide",           "crispr-screen"),
        ("comparative genomics phylogenomics evolution",            "phylogenomics"),
    ],

    # ─────────────────────────────────────────────────────────────────────────
    MATERIALS_SCIENCE: [
        # Nanomaterials & low-dimensional systems
        ("carbon nanotube synthesis properties applications",       "mat-cnt"),
        ("graphene two-dimensional materials electronic properties","mat-graphene"),
        ("nanoparticle synthesis characterization colloidal",       "mat-nanoparticle"),
        ("MXene two-dimensional transition metal carbide",          "mat-mxene"),
        # Energy materials
        ("perovskite solar cell photovoltaic efficiency stability", "mat-perovskite"),
        ("lithium ion battery cathode material electrochemistry",   "mat-battery-cathode"),
        ("solid state electrolyte lithium metal battery",           "mat-solid-electrolyte"),
        ("thermoelectric materials figure of merit ZT",             "mat-thermoelectric"),
        ("solid oxide fuel cell electrode materials",               "mat-sofc"),
        ("hydrogen storage metal hydride materials",                "mat-hydrogen"),
        # Computational materials science
        ("density functional theory first principles materials",    "mat-dft"),
        ("molecular dynamics simulation materials LAMMPS",          "mat-md"),
        ("machine learning interatomic potential materials",        "mat-mlip"),
        ("machine learning materials discovery property prediction","mat-ml-discovery"),
        ("high throughput computational materials screening",       "mat-highthroughput"),
        ("phase field modeling microstructure evolution",           "mat-phasefield"),
        # Structural & metallurgical
        ("high entropy alloy mechanical properties microstructure", "mat-hea"),
        ("additive manufacturing metal 3D printing microstructure", "mat-am"),
        ("grain boundary engineering polycrystalline deformation",  "mat-grainboundary"),
        ("fatigue fracture mechanics structural materials",         "mat-fatigue"),
        ("shape memory alloy martensitic phase transformation",     "mat-sma"),
        # Functional & surface
        ("metal organic framework gas adsorption separation",       "mat-mof"),
        ("heterogeneous catalysis catalyst surface active site",    "mat-catalysis"),
        ("thin film deposition atomic layer deposition growth",     "mat-thinfilm"),
        ("corrosion protection coating surface treatment",          "mat-corrosion"),
        ("polymer composite mechanical characterization reinforcement","mat-composite"),
        ("piezoelectric ferroelectric materials domain switching",  "mat-ferroelectric"),
        ("superconducting materials critical temperature",          "mat-superconductor"),
        ("wide bandgap semiconductor gallium nitride device",       "mat-semiconductor"),
        # Characterization
        ("X-ray diffraction crystal structure refinement",          "mat-xrd"),
        ("transmission electron microscopy materials characterization","mat-tem"),
        ("neutron scattering materials structure dynamics",         "mat-neutron"),
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # Skewed toward clinical, population, and health-services research so the
    # author community stays distinguishable from the bioinformatics cache.
    HEALTH_SCIENCE: [
        # Study design & evidence
        ("randomized controlled trial design methodology",          "hs-rct"),
        ("prospective cohort study risk factor epidemiology",       "hs-cohort"),
        ("systematic review meta-analysis clinical evidence",       "hs-metaanalysis"),
        ("causal inference observational health data",              "hs-causal"),
        # Clinical informatics & prediction
        ("electronic health record clinical informatics",           "hs-ehr"),
        ("machine learning clinical risk prediction model",         "hs-ml-prediction"),
        ("deep learning medical imaging diagnosis",                 "hs-medimaging"),
        ("clinical decision support system implementation",         "hs-cdss"),
        ("natural language processing clinical notes",              "hs-clinical-nlp"),
        ("wearable sensor remote patient monitoring",               "hs-wearable"),
        # Disease areas
        ("cardiovascular disease risk prediction prevention",       "hs-cardio"),
        ("type 2 diabetes management glycemic control",             "hs-diabetes"),
        ("cancer immunotherapy clinical outcomes survival",         "hs-oncology"),
        ("chronic kidney disease progression outcomes",             "hs-ckd"),
        ("stroke rehabilitation functional recovery",               "hs-stroke"),
        ("chronic obstructive pulmonary disease management",        "hs-copd"),
        # Population & public health
        ("infectious disease outbreak epidemic modeling",           "hs-epimodel"),
        ("vaccine effectiveness immunization coverage",             "hs-vaccine"),
        ("health disparities social determinants of health",        "hs-disparities"),
        ("global health low middle income countries",               "hs-globalhealth"),
        ("antimicrobial resistance stewardship surveillance",       "hs-amr"),
        ("maternal child health birth outcomes",                    "hs-maternal"),
        ("environmental exposure air pollution health effects",     "hs-environmental"),
        # Health services, behaviour, outcomes
        ("health economics cost effectiveness analysis",            "hs-healthecon"),
        ("patient reported outcome measures quality of life",       "hs-pro"),
        ("mental health depression anxiety intervention",           "hs-mentalhealth"),
        ("medication adherence pharmacovigilance adverse events",   "hs-pharmacovig"),
        ("nursing care quality patient safety outcomes",            "hs-nursing"),
        ("telemedicine digital health intervention",                "hs-telehealth"),
        ("primary care chronic disease management",                 "hs-primarycare"),
        ("aging geriatrics frailty functional decline",             "hs-aging"),
        ("palliative care end of life outcomes",                    "hs-palliative"),
        ("precision medicine genomic medicine clinical implementation","hs-precisionmed"),
    ],
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def normalize(domain: str | None) -> str:
    """Coerce arbitrary input to a known domain key, falling back to default."""
    if not domain:
        return DEFAULT_DOMAIN
    key = str(domain).strip().lower().replace(" ", "_").replace("-", "_")
    return key if key in DOMAIN_TOPICS else DEFAULT_DOMAIN


def label(domain: str | None) -> str:
    """Human-readable label for a domain key."""
    return DOMAIN_LABELS.get(normalize(domain), DOMAIN_LABELS[DEFAULT_DOMAIN])


def topics_for(domain: str) -> list[tuple[str, str]]:
    return DOMAIN_TOPICS.get(normalize(domain), [])


def flag_key(domain: str) -> str:
    """ChromaDB metadata key marking membership of a domain."""
    return f"dom_{normalize(domain)}"


# ── Topic relevance scoring ──────────────────────────────────────────────────
# Shared by the cache builder's pruning and the agent's read-time paper
# ranking so both order papers identically.
import re as _re  # noqa: E402


def keywords_of(text: str) -> set[str]:
    """Content words usable for lexical overlap scoring."""
    return {w for w in _re.findall(r"[a-z0-9]{4,}", (text or "").lower())}


def topic_relevance(title: str, abstract: str, keywords: set[str],
                    citations: int) -> tuple:
    """
    Rank key for "how well does this paper match these topic keywords".

    Titles are weighted double because they are always present, while abstracts
    are frequently truncated or missing entirely in the Semantic Scholar cache.
    Scoring on the concatenated text lets abstract *length* stand in for
    relevance — a 161-character abstract loses to an 1,800-character one on a
    paper about the same subject. Citations break ties only.
    """
    t = (title or "").lower()
    a = (abstract or "").lower()
    title_overlap    = sum(1 for kw in keywords if kw in t)
    abstract_overlap = sum(1 for kw in keywords if kw in a)
    return (2 * title_overlap + abstract_overlap, citations or 0)
