"""
collect_tool_requirements.py — sourced resource requirements for scientific tools.

Pulls official documentation and pipeline resource configs and keeps the lines
that state hardware needs (memory, CPUs, GPUs, disk), verbatim and with their
source URL, so the Cloud agent cites real requirements instead of generated text.

Output
  cloud_agent/knowledge_base/sourced_tool_requirements.md

Usage
  python journal_fixes/scripts/collect_tool_requirements.py
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cloud_agent" / "knowledge_base" / "sourced_tool_requirements.md"

# (tool, domain, url, what the source is)
SOURCES = [
    ("nf-core/rnaseq", "bioinformatics", "https://raw.githubusercontent.com/nf-core/rnaseq/master/conf/base.config", "resource labels used by every process (official pipeline config)"),
    ("nf-core/rnaseq", "bioinformatics", "https://raw.githubusercontent.com/nf-core/rnaseq/master/docs/usage.md", "official usage documentation"),
    ("nf-core/sarek (variant calling, GATK)", "bioinformatics", "https://raw.githubusercontent.com/nf-core/sarek/master/conf/base.config", "resource labels used by every process (official pipeline config)"),
    ("nf-core/scrnaseq", "bioinformatics", "https://raw.githubusercontent.com/nf-core/scrnaseq/master/conf/base.config", "resource labels used by every process (official pipeline config)"),
    ("nf-core/atacseq", "bioinformatics", "https://raw.githubusercontent.com/nf-core/atacseq/master/conf/base.config", "resource labels used by every process (official pipeline config)"),
    ("nf-core/mag (metagenomics)", "bioinformatics", "https://raw.githubusercontent.com/nf-core/mag/master/conf/base.config", "resource labels used by every process (official pipeline config)"),
    ("AlphaFold 2", "bioinformatics", "https://raw.githubusercontent.com/google-deepmind/alphafold/main/README.md", "official README"),
    ("ColabFold", "bioinformatics", "https://raw.githubusercontent.com/sokrypton/ColabFold/main/README.md", "official README"),
    ("BWA-MEM2", "bioinformatics", "https://raw.githubusercontent.com/bwa-mem2/bwa-mem2/master/README.md", "official README"),
    ("minimap2", "bioinformatics", "https://raw.githubusercontent.com/lh3/minimap2/master/README.md", "official README"),
    ("Kraken 2", "bioinformatics", "https://raw.githubusercontent.com/DerrickWood/kraken2/master/docs/MANUAL.markdown", "official manual"),
    ("DeepVariant", "bioinformatics", "https://raw.githubusercontent.com/google/deepvariant/r1.6/docs/deepvariant-details.md", "official documentation"),
    ("Salmon", "bioinformatics", "https://raw.githubusercontent.com/COMBINE-lab/salmon/master/README.md", "official README"),
    ("nnU-Net", "health_science", "https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/documentation/installation_instructions.md", "official installation documentation"),
    ("MONAI", "health_science", "https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/README.md", "official README"),
    ("LAMMPS (GPU package)", "materials_science", "https://raw.githubusercontent.com/lammps/lammps/develop/doc/src/Speed_gpu.rst", "official documentation"),
    ("Quantum ESPRESSO", "materials_science", "https://raw.githubusercontent.com/QEF/q-e/develop/README.md", "official README"),
    ("MACE (ML interatomic potential)", "materials_science", "https://raw.githubusercontent.com/ACEsuit/mace/main/README.md", "official README"),
]

KEY = re.compile(r"(\b\d+(\.\d+)?\s?(GB|GiB|TB|TiB|MB|G\b|cpus?|cores?|threads?)\b|memory|RAM\b|GPU|VRAM|CUDA|disk space|cpus\s*=|memory\s*=|time\s*=)", re.I)


def main() -> None:
    lines = [
        "# Sourced resource requirements for scientific tools",
        "",
        f"Every statement below is quoted verbatim from the linked official source, retrieved {date.today().isoformat()}.",
        "Use these before extrapolating. nf-core `base.config` labels (process_low/medium/high) are the defaults the pipeline requests for each step.",
        "",
    ]
    for tool, domain, url, kind in SOURCES:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except Exception as exc:
            print(f"[skip] {tool}: {exc}")
            continue
        src = r.text.splitlines()
        keep, seen = [], set()
        for i, ln in enumerate(src):
            if KEY.search(ln) and len(ln.strip()) > 3:
                # keep a little context for config blocks (label name lines)
                lo = i - 1 if i > 0 and ("withLabel" in src[i - 1] or "withName" in src[i - 1]) else i
                for j in range(lo, i + 1):
                    t = src[j].rstrip()
                    if t.strip() and j not in seen and len(t) < 400:
                        keep.append(t)
                        seen.add(j)
        if not keep:
            print(f"[no matches] {tool}")
            continue
        keep = keep[:60]
        lines += [f"## {tool} ({domain})", "", f"Source: {kind}, {url}", "", "```text", *keep, "```", ""]
        print(f"{tool:40s} {len(keep):3d} lines")
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
