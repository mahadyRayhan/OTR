"""
collect_bio_bench.py — the documents that the bioinformatics test questions need, and nothing else.

The existing knowledge bases stay as they are and act as the haystack. This script only adds the
source documents of the public test questions, written to a separate folder so the deployed
knowledge bases are not changed:

  ~/otr_local/bench_kb/bio/
    rag/litqa2/<PMCID>.txt      full text of each LitQA2 source paper (Europe PMC open access)
    rag/bioasq/<PMID>.txt       gold abstracts of the sampled BioASQ questions (from the BioASQ file)
    code/repos/<owner__repo>/   BioCoder source repositories, with every target function removed
    cloud/tool_docs/<tool>.md   README of each tool in the cloud and full-system sets (GitHub homepages)
    cloud/nfcore_analogues.md   resource labels of nf-core modules whose tools are NOT in those sets
    manifest.json               what was fetched, from where, and what failed

It also writes journal_fixes/data/testsets/bio/ : the bioinformatics test sets, restricted to items
whose documents could be obtained.

No LLM calls. Usage: python journal_fixes/scripts/collect_bio_bench.py [--step all|litqa2|bioasq|biocoder|tooldocs|analogues|testsets]
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from collect_oa_papers import API, fetch_fulltext, session  # noqa: E402

LOCAL = Path.home() / "otr_local"
BENCH = LOCAL / "benchmarks"
NFCORE = LOCAL / "nfcore_modules" / "modules" / "nf-core"
OUT = LOCAL / "bench_kb" / "bio"
TESTSETS = ROOT / "journal_fixes" / "data" / "testsets"
SEED = 0
# BioASQ questions mentioning any of these are left for the medical domain.
CLINICAL = re.compile(r"\b(patient|patients|clinical|therapy|therapies|treatment|treated|drug|drugs|dose|"
                      r"symptom|symptoms|diagnos\w*|syndrome|disease|diseases|disorder|cancer|tumou?r|"
                      r"carcinoma|infection|surgery|mortality|trial|prognos\w*|medication|vaccine|"
                      r"inhibitors?|blockers?|agonists?|antagonists?|sepsis|melanoma|leukemia|lymphoma|"
                      r"outcomes?|survival|risk|efficacy|approved|FDA|\w+(?:mab|nib|stat|vir|cillin))\b", re.I)


def manifest() -> dict:
    p = OUT / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_manifest(m: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(m, indent=1))


# ── RAG: LitQA2 source papers ────────────────────────────────────────────────

def litqa2() -> None:
    items = json.loads((ROOT / "journal_fixes/data/benchmarks/labbench_LitQA2.json").read_text())
    out = OUT / "rag" / "litqa2"
    out.mkdir(parents=True, exist_ok=True)
    m = manifest()
    got = m.setdefault("litqa2", {})
    dois = sorted({s.lower().replace("https://doi.org/", "") for x in items for s in x.get("sources") or []})
    for doi in dois:
        if doi in got:
            continue
        rec = {"doi": doi, "status": "not found"}
        try:
            r = session.get(f"{API}/search", params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"},
                            timeout=60)
            hits = r.json().get("resultList", {}).get("result", [])
            h = hits[0] if hits else None
            if h:
                rec.update(pmcid=h.get("pmcid"), pmid=h.get("pmid"), title=h.get("title"), year=h.get("pubYear"),
                           license=h.get("license"), journal=h.get("journalTitle"), status="no open-access full text")
                if h.get("pmcid") and h.get("isOpenAccess") == "Y":
                    ft = fetch_fulltext(h["pmcid"])
                    if ft:
                        abstract, body = ft
                        header = "\n".join(f"{k}: {rec.get(k)}" for k in ("pmcid", "pmid", "doi", "title", "journal", "year", "license"))
                        (out / f'{h["pmcid"]}.txt').write_text(
                            f'# {rec["title"]}\n{header}\ndomain: bioinformatics\nurl: https://europepmc.org/article/PMC/{h["pmcid"]}\n\n'
                            f"## Abstract\n{abstract}\n\n{body}\n", encoding="utf-8")
                        rec["status"] = "full text"
        except Exception as exc:
            rec["status"] = f"error: {exc!r}"[:200]
        got[doi] = rec
        time.sleep(0.2)
    # Fallbacks for papers outside the Europe PMC open-access subset.
    for doi, rec in got.items():
        if rec["status"] == "full text":
            continue
        try:
            text, source = None, None
            if rec.get("pmcid"):
                text, source = _bioc_fulltext(rec["pmcid"]), "NCBI BioC (PMC open access / author manuscript)"
            if not text and doi.startswith("10.1101/"):
                text, source, lic = _biorxiv_fulltext(doi)
                if text:
                    rec["license"] = lic
            if text:
                name = rec.get("pmcid") or "biorxiv_" + re.sub(r"\W", "_", doi)
                header = "\n".join(f"{k}: {rec.get(k)}" for k in ("pmcid", "pmid", "doi", "title", "journal", "year", "license"))
                (out / f"{name}.txt").write_text(f'# {rec.get("title")}\n{header}\ndomain: bioinformatics\nsource: {source}\n\n{text}\n',
                                                 encoding="utf-8")
                rec.update(status="full text", source=source)
        except Exception as exc:
            rec["fallback_error"] = repr(exc)[:200]
        time.sleep(8 if doi.startswith("10.1101/") else 0.3)
    save_manifest(m)
    print("LitQA2 sources:", collections.Counter(v["status"] for v in got.values()))


def _bioc_fulltext(pmcid: str) -> str | None:
    """Full text from NCBI's BioC API, which covers the PMC open-access subset and author manuscripts."""
    r = session.get(f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode", timeout=60)
    if r.status_code != 200 or not r.text.strip().startswith(("[", "{")):
        return None
    data = r.json()
    docs = data if isinstance(data, list) else [data]
    paras = []
    for coll in docs:
        for doc in coll.get("documents", []):
            for p in doc.get("passages", []):
                sec = p.get("infons", {}).get("section_type", "")
                if sec in ("REF", "COMP_INT", "AUTH_CONT", "ACK_FUND") or not p.get("text"):
                    continue
                paras.append(p["text"])
    text = "\n".join(paras)
    return text if len(text.split()) > 1500 else None


def _biorxiv_fulltext(doi: str) -> tuple[str | None, str | None, str | None]:
    """Full text of a bioRxiv preprint from the JATS XML that the bioRxiv API links to (latest version)."""
    import xml.etree.ElementTree as ET
    r = session.get(f"https://api.biorxiv.org/details/biorxiv/{doi}", timeout=60)
    coll = r.json().get("collection", []) if r.status_code == 200 else []
    if not coll:
        return None, None, None
    last = coll[-1]
    for wait in (0, 60, 180):  # biorxiv.org rate-limits bulk downloads (HTTP 429, Cloudflare 1015)
        time.sleep(wait)
        x = session.get(last["jatsxml"], timeout=90)
        if x.status_code != 429:
            break
    if x.status_code != 200:
        return None, None, None
    root = ET.fromstring(x.content)
    flat = lambda el: re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""
    body = root.find(".//body")
    text = f"## Abstract\n{flat(root.find('.//abstract'))}\n\n" + "\n".join(flat(p) for p in body.iter("p")) if body is not None else ""
    return (text if len(text.split()) > 1500 else None), "bioRxiv JATS XML", last.get("license")


# ── RAG: BioASQ bio questions and their gold abstracts ───────────────────────

def bioasq(n: int = 150) -> None:
    import pandas as pd
    qs = json.loads((BENCH / "bioasq_training11b.json").read_text())["questions"]
    passages = pd.read_parquet(BENCH / "bioasq_mini_passages.parquet")["passage"].to_dict()
    pmid_of = lambda d: d.rstrip("/").split("/")[-1]
    pool = []
    for q in qs:
        if q["type"] not in ("yesno", "factoid") or CLINICAL.search(q["body"]):
            continue
        gold = [pmid_of(d) for d in q.get("documents", []) if int(pmid_of(d)) in passages]
        if gold:
            pool.append({**q, "gold_pmids": gold})
    rng = random.Random(SEED)
    by_type = collections.defaultdict(list)
    for q in pool:
        by_type[q["type"]].append(q)
    picked = []
    for t, g in sorted(by_type.items()):
        rng.shuffle(g)
        picked += g[:round(n * len(g) / len(pool))]
    out = OUT / "rag" / "bioasq"
    out.mkdir(parents=True, exist_ok=True)
    for q in picked:
        for p in q["gold_pmids"]:
            (out / f"{p}.txt").write_text(f"# PubMed {p}\npmid: {p}\ndomain: bioinformatics\nurl: https://pubmed.ncbi.nlm.nih.gov/{p}/\n\n"
                                          f"## Abstract\n{passages[int(p)]}\n", encoding="utf-8")
    (OUT / "bioasq_selected.json").write_text(json.dumps(picked, indent=1))
    m = manifest()
    m["bioasq"] = {"pool_bio_yesno_factoid": len(pool), "selected": len(picked),
                   "types": dict(collections.Counter(q["type"] for q in picked)),
                   "gold_abstracts": len({p for q in picked for p in q["gold_pmids"]})}
    save_manifest(m)
    print("BioASQ:", m["bioasq"])


# ── Code: BioCoder repositories with the target functions removed ────────────

def _remove_functions(path: Path, names: set[str]) -> int:
    src = path.read_text(errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    lines = src.splitlines(keepends=True)
    spans = [(n.lineno - len(n.decorator_list), n.end_lineno) for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    for start, end in sorted(spans, reverse=True):
        indent = re.match(r"\s*", lines[start - 1]).group(0)
        lines[start - 1:end] = [f"{indent}# [removed for evaluation: target function of a BioCoder test item]\n"]
    path.write_text("".join(lines))
    return len(spans)


def _residual_copies(repo_dir: Path, items: list[dict], n: int = 20, threshold: float = 0.2) -> list[str]:
    """Items whose function body (as code-token 20-grams) is still >20% present elsewhere in the repository,
    e.g. a duplicate copy in another file. These items are excluded from the test set."""
    tok = lambda s: re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", s)

    def grams(s):
        t = tok(s)
        return {" ".join(t[i:i + n]) for i in range(len(t) - n + 1)}

    repo = set().union(*(grams(f.read_text(errors="ignore")) for f in repo_dir.rglob("*.py")))
    out = []
    for x in items:
        body = grams(x["goldenCode"].split("\n", 1)[-1])
        if body and len(body & repo) / len(body) > threshold:
            out.append(x["test_case_id"])
    return out


def biocoder() -> None:
    items = [x for x in json.loads((BENCH / "biocoder.json").read_text()) if x["language"] == "Python"]
    by_repo = collections.defaultdict(list)
    for x in items:
        by_repo[x["repository"]].append(x)
    base = OUT / "code" / "repos"
    base.mkdir(parents=True, exist_ok=True)
    m = manifest()
    rep = m.setdefault("biocoder", {})
    for repo, xs in sorted(by_repo.items()):
        d = base / repo.replace("/", "__")
        if not d.exists():
            r = subprocess.run(["git", "clone", "-q", "--depth", "1", f"https://github.com/{repo}.git", str(d)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                rep[repo] = {"status": "clone failed", "items": len(xs)}
                continue
            subprocess.run(["rm", "-rf", str(d / ".git")])
            for f in d.rglob("*"):  # keep source and docs only
                if f.is_file() and f.suffix not in (".py", ".md", ".rst", ".txt"):
                    f.unlink()
        removed, missing = 0, []
        for x in xs:
            name = re.search(r"\bdef\s+(\w+)", x["signature"]).group(1)  # signatures may start with decorators
            f = d / x["filePath"]
            k = _remove_functions(f, {name}) if f.exists() else 0
            removed += k
            if not k:
                missing.append(x["test_case_id"])
        rep[repo] = {"status": "ok", "items": len(xs), "functions_removed": removed,
                     "items_without_function_in_head": missing, "py_files": sum(1 for _ in d.rglob("*.py")),
                     "items_body_still_in_repo": _residual_copies(d, [x for x in xs if x["test_case_id"] not in missing])}
    save_manifest(m)
    print("BioCoder repos:", {k: (v["status"], v.get("functions_removed"), len(v.get("items_without_function_in_head", [])))
                              for k, v in rep.items()})


# ── Cloud / full system: tool READMEs and nf-core analogues ──────────────────

def _module_meta(module: str) -> dict:
    return yaml.safe_load((NFCORE / module / "meta.yml").read_text())


def tooldocs() -> None:
    tools = {}
    for f in (TESTSETS / "per_agent" / "cloud_nfcore.jsonl", TESTSETS / "full_system_nfcore.jsonl"):
        for x in map(json.loads, open(f)):
            tools[x["meta"]["tool"]] = x["meta"]["module"]
    out = OUT / "cloud" / "tool_docs"
    out.mkdir(parents=True, exist_ok=True)
    m = manifest()
    rep = m.setdefault("tool_docs", {})
    for tool, module in sorted(tools.items()):
        if tool in rep and rep[tool]["status"] == "ok":
            continue
        info = next(iter(_module_meta(module)["tools"][0].values())) or {}
        urls = [str(info.get(k) or "") for k in ("homepage", "documentation", "tool_dev_url")]
        gh = next((re.match(r"https?://github\.com/([^/\s]+/[^/\s#?]+)", u) for u in urls if "github.com" in u), None)
        rec = {"status": "no GitHub homepage", "homepage": urls[0]}
        if gh:
            slug = gh.group(1).removesuffix(".git")
            for name in ("README.md", "README.rst", "README", "readme.md", "README.txt"):
                try:
                    r = session.get(f"https://raw.githubusercontent.com/{slug}/HEAD/{name}", timeout=30)
                except requests.RequestException:
                    continue
                if r.status_code == 200 and len(r.text) > 200:
                    (out / f"{tool}.md").write_text(f"# {tool}: README\nsource: https://github.com/{slug}\n\n{r.text}", encoding="utf-8")
                    rec = {"status": "ok", "source": f"https://github.com/{slug}/blob/HEAD/{name}"}
                    break
            else:
                rec = {"status": "README not found", "repo": slug}
            time.sleep(0.1)
        rep[tool] = rec
        if len(rep) % 25 == 0:
            save_manifest(m)
    save_manifest(m)
    print("tool docs:", collections.Counter(v["status"] for v in rep.values()))


def analogues() -> None:
    """Resource labels of modules whose tools are NOT tested, so the cloud agent can reason from similar steps.
    DOIs are left out (they are scholar gold)."""
    tested = set()
    for f in (TESTSETS / "per_agent" / "cloud_nfcore.jsonl", TESTSETS / "full_system_nfcore.jsonl"):
        tested |= {x["meta"]["tool"] for x in map(json.loads, open(f))}
    rows = []
    for meta_f in sorted(NFCORE.rglob("meta.yml")):
        module = str(meta_f.parent.relative_to(NFCORE))
        main = meta_f.parent / "main.nf"
        if not main.exists():
            continue
        try:
            meta = yaml.safe_load(meta_f.read_text())
            tool = next(iter(meta["tools"][0])) if meta.get("tools") else None
        except Exception:
            continue
        if not tool or tool in tested:
            continue
        labels = sorted(set(re.findall(r'label\s+[\'"](process_\w+)[\'"]', main.read_text(errors="ignore"))))
        if labels:
            rows.append(f'| {module} | {tool} | {" ".join(str(meta.get("description") or "").split())[:160]} | {", ".join(labels)} |')
    out = OUT / "cloud"
    out.mkdir(parents=True, exist_ok=True)
    (out / "nfcore_analogues.md").write_text(
        "# nf-core modules and their default resource labels\n"
        "source: https://github.com/nf-core/modules (MIT). Label defaults: nf-core pipeline template conf/base.config\n\n"
        "| module | tool | description | labels |\n| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n", encoding="utf-8")
    m = manifest()
    m["nfcore_analogues"] = {"modules": len(rows), "excluded_tested_tools": len(tested)}
    save_manifest(m)
    print("analogue modules:", len(rows))


# ── Bio test sets restricted to obtainable documents ─────────────────────────

def testsets() -> None:
    m = manifest()
    out = TESTSETS / "bio"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    ok_dois = {d for d, v in m["litqa2"].items() if v["status"] == "full text"}
    summary = {}

    def write(name, items):
        with open(out / f"{name}.jsonl", "w") as f:
            for x in items:
                f.write(json.dumps(x) + "\n")
        summary[name] = {"n": len(items), "dev": sum(x["split"] == "dev" for x in items),
                         "test": sum(x["split"] == "test" for x in items)}

    def split(items):
        items = items[:]
        rng.shuffle(items)
        for i, x in enumerate(items):
            x["split"] = "dev" if i < round(0.2 * len(items)) else "test"
        return items

    # LitQA2: every item whose source paper text was obtained (the 2 earlier leakage items included:
    # in this setting the source papers are in the corpus by design).
    from build_testsets import labbench, scholar_claims
    lit = labbench("LitQA2", "rag", set(), rng)
    lit = [x for x in lit if any(s.lower().replace("https://doi.org/", "") in ok_dois for s in x["meta"]["sources"] or [])]
    write("rag_litqa2", split(lit))
    claims = [x for x in scholar_claims(set()) if set(x["gold"]) & ok_dois]
    write("scholar_claimpaper", split(claims))

    sel = json.loads((OUT / "bioasq_selected.json").read_text())
    write("rag_bioasq", split([{
        "id": q["id"], "benchmark": "bioasq_11b", "agent": "rag", "question": q["body"],
        "answer_format": q["type"], "gold": q["exact_answer"],
        "scoring": "yes/no exact; factoid: any synonym of the gold entity in the answer (case-insensitive)",
        "meta": {"gold_pmids": q["gold_pmids"], "ideal_answer": q.get("ideal_answer")}} for q in sel]))

    bc_ok = {r for r, v in m["biocoder"].items() if v["status"] == "ok"}
    bc_missing = {i for v in m["biocoder"].values()
                  for i in v.get("items_without_function_in_head", []) + v.get("items_body_still_in_repo", [])}
    bc = [json.loads(l) for l in open(TESTSETS / "per_agent" / "code_biocoder.jsonl")]
    write("code_biocoder", [x for x in bc if x["meta"]["repository"] in bc_ok and x["id"] not in bc_missing])
    for name in ("code_seqqa", "cloud_nfcore", "scholar_toolpaper"):
        write(name, [json.loads(l) for l in open(TESTSETS / "per_agent" / f"{name}.jsonl")])
    full = [json.loads(l) for l in open(TESTSETS / "full_system_nfcore.jsonl")]
    docs = {t for t, v in m["tool_docs"].items() if v["status"] == "ok"}
    for x in full:
        x["meta"]["tool_readme_in_kb"] = x["meta"]["tool"] in docs
    write("full_system", full)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all",
                    choices=["all", "litqa2", "bioasq", "biocoder", "tooldocs", "analogues", "testsets"])
    s = ap.parse_args().step
    steps = {"litqa2": litqa2, "bioasq": bioasq, "biocoder": biocoder, "tooldocs": tooldocs,
             "analogues": analogues, "testsets": testsets}
    for name, fn in steps.items():
        if s in ("all", name):
            fn()
