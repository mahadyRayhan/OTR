"""
build_multipart_pilot.py — 10 multi-part research questions and their 30 single-part versions, with checklists.

The questions come from the 50-query benchmark (scripts/run_ablation_benchmark.py BENCHMARK_QUERIES), drawn at
random within each topic (not the 30 "best" queries, which were chosen because SciWeaver won on them). Each has
three parts — explain (rag), code, cloud — and is split mechanically into three single-part questions; the cloud
part carries the code sentence as its workload, so it stands alone.

Each item's gold is a checklist scored by scoring.py (format "checklist"), no LLM:
  doi       the answer cites one of the key papers (DOIs checked against Crossref with --verify)
  terms     each group of patterns must match somewhere in the answer (key concepts, AWS components)
  parses    a ```python block parses (Snakemake: at least three rules)
  calls     each group of patterns must match inside the code blocks (the functions and settings asked for)
Score: credit = share of checks passed; correct = all passed; per section (explain / code / cloud).

Writes journal_fixes/data/testsets/<domain>/multipart_pilot.jsonl (10) and multipart_parts.jsonl (30), split "dev".
Usage: python journal_fixes/scripts/build_multipart_pilot.py [--verify]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from journal_fixes.scripts.build_testsets import item  # noqa: E402

PICKED = [5, 7, 10, 35, 36, 41, 42, 44, 49, 50]  # see REPORT.md: seeded draw per topic, analysis-workflow topics

DOI = {
    "leiden": "10.1038/s41598-019-41695-z", "squidpy": "10.1038/s41592-021-01358-2",
    "deseq2": "10.1186/s13059-014-0550-8", "indep_filter": "10.1073/pnas.0914005107",
    "star": "10.1093/bioinformatics/bts635", "featurecounts": "10.1093/bioinformatics/btt656",
    "snakemake": "10.1093/bioinformatics/bts480", "snakemake2021": "10.12688/f1000research.29032.2",
    "wnn": "10.1016/j.cell.2021.04.048", "hnsw": "10.1109/TPAMI.2018.2889473",
    "muon": "10.1186/s13059-021-02577-8", "citeseq": "10.1038/nmeth.4380",
    "starfusion": "10.1186/s13059-019-1842-9", "sam": "10.1093/bioinformatics/btp352",
}

# per question: explain (dois, term groups), code (python | snakemake, call groups), cloud (term groups)
SPEC = {
    5: {"dois": ["leiden"], "explain": [r"well[- ]connected", r"disconnect", r"refine"],
        "code": [r"leidenalg", r"planted.?partition|stochastic.?block|sbm", r"louvain", r"internal|external|cut"],
        "cloud": [r"lambda", r"layer|container image|zip|deployment package"]},
    7: {"dois": ["leiden"], "explain": [r"aggregat", r"refine", r"linear|scal|million"],
        "code": [r"leiden", r"resolution\s*=\s*0\.4", r"tracemalloc", r"neighbors\(", r"umap",
                 r"dpi\s*=\s*[3-9]\d\d|\.pdf|\.svg|\.tiff?"],
        "cloud": [r"r6i\.8xlarge", r"\befs\b|elastic file system", r"\biam\b|access point|posix|multi[- ]user"]},
    10: {"dois": ["leiden", "squidpy"], "explain": [r"spatial", r"neighbo(u)?r graph|delaunay|radius|k-?nearest|knn"],
         "code": [r"squidpy|\bsq\.", r"spatial_neighbors", r"leiden", r"resolution\s*=\s*0\.4",
                  r"spatial_scatter|pl\.spatial"],
         "cloud": [r"r6i\.4xlarge", r"\bs3\b"]},
    35: {"dois": ["deseq2", "indep_filter"],
         "explain": [r"independent filtering", r"mean (of )?normali[sz]ed count|basemean", r"benjamini|\bbh\b|false discovery"],
         "code": [r"pydeseq2", r"DeseqDataSet|DeseqStats", r"\b20\b", r"multipletests|fdr|benjamini|false_discovery|padj",
                  r"savefig|plt\.|matplotlib"],
         "cloud": [r"r6i\.xlarge"]},
    36: {"dois": ["deseq2", "star", "featurecounts", "snakemake", "snakemake2021"],
         "explain": [r"\bSTAR\b", r"featureCounts", r"snakemake"], "snakemake": True,
         "code": [r"\bSTAR\b", r"twopassMode|two.?pass", r"featureCounts", r"pydeseq2", r"volcano"],
         "cloud": [r"\bbatch\b", r"\bspot\b", r"fsx|lustre"]},
    41: {"dois": ["wnn"], "explain": [r"concatenat", r"weight"],
         "code": [r"leiden", r"adjusted_rand_score|\bARI\b", r"umap", r"wnn|weighted.?nearest|mu\.pp\.neighbors|muon",
                  r"protein|\badt\b|prot"],
         "cloud": [r"r6i\.4xlarge", r"\bs3\b"]},
    42: {"dois": ["wnn", "muon", "citeseq"], "explain": [r"protein", r"rare", r"annotat"],
         "code": [r"muon|\bmu\.", r"CD3E", r"CD19", r"CD14", r"CD56|NCAM1", r"rank_genes_groups", r"dotplot"],
         "cloud": [r"sagemaker", r"jupyterlab|studio|notebook"]},
    44: {"dois": ["wnn", "hnsw"], "explain": [r"hnsw|hierarchical navigable", r"approximate"],
         "code": [r"100[_,]?000|1e5", r"n_neighbors\s*=\s*30", r"tracemalloc", r"leiden", r"perf_counter|time\.time|timeit",
                  r"pynndescent|hnswlib|annoy|approx|knn_method|transformer"],
         "cloud": [r"r6i\.8xlarge", r"gp3", r"\bebs\b"]},
    49: {"dois": ["star", "starfusion"], "explain": [r"chimSegmentMin", r"Chimeric\.out\.junction", r"fusion"],
         "code": [r"subprocess", r"chimSegmentMin\W+20", r"Chimeric\.out\.junction", r"(>=|>)\s*[45]\b",
                  r"100[_,]?000|1e5|100\s*\*\s*1000", r"\.bed"],
         "cloud": [r"\bbatch\b", r"\bs3\b"]},
    50: {"dois": ["sam", "star"], "explain": [r"\bNH\b", r"\bHI\b", r"\bAS\b", r"\bNM\b", r"primary|unique"],
         "code": [r"pysam", r"get_tag|has_tag", r"['\"]NH['\"]", r"is_secondary|is_supplementary|==\s*1",
                  r"AlignmentFile\(.*['\"]wb['\"]|template\s*="],
         "cloud": [r"lambda", r"\bs3\b", r"/tmp|ephemeral|layer|container"]},
}


def queries() -> list[str]:
    tree = ast.parse((ROOT / "scripts/run_ablation_benchmark.py").read_text())
    return next(ast.literal_eval(n.value) for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BENCHMARK_QUERIES")


def split_parts(q: str) -> dict[str, str]:
    m = re.match(r"(.*?)\s+(Write a .*?)\s+((?:Design|Deploy) .*)$", q, re.S)
    if not m:
        raise ValueError(f"cannot split: {q[:80]}")
    explain, code, cloud = m.groups()
    return {"explain": explain, "code": code, "cloud": f"{cloud} (The workload to run: {code[0].lower()}{code[1:]})"}


def readable(pattern: str) -> str:
    """A regex as words for the judge's checklist, e.g. r"planted.?partition|sbm" -> "planted partition, or sbm"."""
    for x, y in ((r"\b", ""), (r"\s*", " "), (r"\W+", " "), (".?", " "), (r"\.", "."), ("(?:", "("), ("\\", "")):
        pattern = pattern.replace(x, y)
    return ", or ".join(t.strip() for t in pattern.split("|"))


def checks(spec: dict) -> list[dict]:
    out = [{"id": "explain.doi", "section": "explain", "kind": "doi", "dois": [DOI[k] for k in spec["dois"]],
            "desc": "cites a key paper: " + ", ".join(spec["dois"])}]
    out += [{"id": f"explain.t{i}", "section": "explain", "kind": "terms", "pattern": p, "desc": f"explains: {readable(p)}"}
            for i, p in enumerate(spec["explain"], 1)]
    out.append({"id": "code.parses", "section": "code", "kind": "snakemake" if spec.get("snakemake") else "parses",
                "desc": "a Snakemake workflow with at least three rules" if spec.get("snakemake") else "a Python block that parses"})
    out += [{"id": f"code.c{i}", "section": "code", "kind": "calls", "pattern": p, "desc": f"code uses: {readable(p)}"}
            for i, p in enumerate(spec["code"], 1)]
    out += [{"id": f"cloud.t{i}", "section": "cloud", "kind": "terms", "pattern": p, "desc": f"cloud design has: {readable(p)}"}
            for i, p in enumerate(spec["cloud"], 1)]
    return out


def verify() -> None:
    for k, d in DOI.items():
        try:
            with urllib.request.urlopen(f"https://api.crossref.org/works/{d}", timeout=20) as r:
                print(f"  {k:14} {d:34} {json.loads(r.read())['message']['title'][0][:70]}")
        except Exception as e:
            print(f"  {k:14} {d:34} NOT FOUND ({e})")


def main(domain: str) -> None:
    qs = queries()
    multi, parts = [], []
    for n in PICKED:
        q, ch = qs[n - 1], checks(SPEC[n])
        qid = f"mp_q{n:02d}"
        multi.append({**item(qid, "benchmark50", "full", q, "checklist", {"checks": ch}, "checklist (scoring.py)",
                             query_index=n), "split": "dev"})
        for sec, text in split_parts(q).items():
            parts.append({**item(f"{qid}_{sec}", "benchmark50", {"explain": "rag"}.get(sec, sec), text, "checklist",
                                 {"checks": [c for c in ch if c["section"] == sec]}, "checklist (scoring.py)",
                                 query_index=n, parent=qid, section=sec), "split": "dev"})
    d = ROOT / "journal_fixes/data/testsets" / domain
    (d / "multipart_pilot.jsonl").write_text("".join(json.dumps(x) + "\n" for x in multi))
    (d / "multipart_parts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in parts))
    print(f"multipart_pilot.jsonl: {len(multi)} items, {sum(len(x['gold']['checks']) for x in multi)} checks; "
          f"multipart_parts.jsonl: {len(parts)} items")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bio")
    ap.add_argument("--verify", action="store_true", help="look every DOI up in Crossref")
    a = ap.parse_args()
    if a.verify:
        verify()
    main(a.domain)
