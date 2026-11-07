"""
build_testsets.py — per-agent and full-system test sets from public sources only.

Every item has a gold answer that can be scored without an LLM judge. Items flagged by
leakage_check.py are excluded. Sampling uses a fixed seed; 20% of each set is a dev split
(for tuning tau, prompts and the D1 router), the rest is test.

Per-agent sets (journal_fixes/data/testsets/per_agent/)
  rag_litqa2         LAB-Bench LitQA2 (MCQ)                      CC BY-SA 4.0
  rag_pubmedqa       PubMedQA labelled (yes / no / maybe)         MIT
  rag_matscibench    MatSciBench, numeric, single answer, no image (relative tolerance)
  rag_protocolqa     LAB-Bench ProtocolQA (MCQ)                   CC BY-SA 4.0
  code_scicode       SciCode main problems (official unit tests; its own dev split)  Apache-2.0
  code_biocoder      BioCoder public, Python (official test harness)                  CC BY 4.0
  code_seqqa         LAB-Bench SeqQA (MCQ)                        CC BY-SA 4.0
  cloud_nfcore       nf-core module -> default CPU / memory, cheapest fitting EC2 instance  MIT
  scholar_toolpaper  nf-core module -> DOI of the tool's paper     MIT
  scholar_claimpaper LitQA2 key passage -> DOI of the source paper CC BY-SA 4.0

Full-system set (journal_fixes/data/testsets/full_system_nfcore.jsonl)
  One nf-core step per item, three graded parts that need different agents:
  literature (tool paper DOI), code (command line: executable and subcommand),
  cloud (CPU / memory request and a cheapest-fitting x86 instance).
  Tools are split so that no tool appears in more than one of cloud / scholar / full-system.

No LLM calls. Usage: python journal_fixes/scripts/build_testsets.py [--bench-dir ~/otr_local/benchmarks]
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import random
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "journal_fixes" / "data" / "testsets"
SEED = 0
DEV_FRACTION = 0.2
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# nf-core pipeline template, conf/base.config (https://github.com/nf-core/tools)
NFCORE_LABELS = {"process_single": (1, 6), "process_low": (2, 12), "process_medium": (6, 36),
                 "process_high": (12, 72), "process_high_memory": (12, 200)}
INTERPRETERS = {"python", "python3", "Rscript", "R", "perl", "bash", "sh", "java", "ruby", "node"}
CAMEL_SUBCOMMAND = {"gatk", "gatk3", "picard"}  # subcommands such as FastqToSam follow JVM options
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;,'\"]+")
BIOTOOLS_CACHE = Path.home() / "otr_local" / "biotools_cache.json"
SHELL_SKIP = {"export", "mkdir", "cp", "mv", "ln", "if", "fi", "then", "else", "cat", "echo", "gzip", "gunzip",
              "zcat", "set", "rm", "touch", "tar", "def", "for", "do", "done", "cd", "printf", "sed", "awk",
              "find", "unzip", "bgzip", "tabix", "ls", "while", "test", "[", "[[", "eval", "local", "true"}


def split(items: list[dict], rng: random.Random) -> list[dict]:
    items = items[:]
    rng.shuffle(items)
    n_dev = round(len(items) * DEV_FRACTION)
    for i, x in enumerate(items):
        x["split"] = "dev" if i < n_dev else "test"
    return items


def stratified(items: list[dict], key, n: int, rng: random.Random) -> list[dict]:
    """Proportional sample of n items, stratified by key(item)."""
    groups = collections.defaultdict(list)
    for x in items:
        groups[key(x)].append(x)
    out = []
    for g in groups.values():
        rng.shuffle(g)
        out += g[:max(1, round(n * len(g) / len(items)))]
    rng.shuffle(out)
    return out[:n]


def mcq(x: dict, rng: random.Random) -> tuple[list[str], str]:
    opts = [x["ideal"], *x["distractors"]]
    rng.shuffle(opts)
    return opts, LETTERS[opts.index(x["ideal"])]


def item(id_, bench, agent, question, fmt, gold, scoring, **meta) -> dict:
    return {"id": id_, "benchmark": bench, "agent": agent, "question": question, "answer_format": fmt,
            "gold": gold, "scoring": scoring, "meta": meta}


# ── per-agent sets from existing benchmarks ──────────────────────────────────

def labbench(sub: str, agent: str, flags: set, rng: random.Random, n: int | None = None) -> list[dict]:
    data = [x for x in json.loads((ROOT / f"journal_fixes/data/benchmarks/labbench_{sub}.json").read_text())
            if x["id"] not in flags]
    if n:
        rng.shuffle(data)
        data = data[:n]
    out = []
    for x in data:
        opts, letter = mcq(x, rng)
        out.append(item(x["id"], f"labbench_{sub}", agent, x["question"], "mcq", letter,
                        "exact letter; an 'insufficient information' option may be added for precision / coverage",
                        choices=opts, sources=x.get("sources"), canary=x.get("canary")))
    return out


def pubmedqa(bench: Path, flags: set, rng: random.Random, n: int) -> list[dict]:
    import pandas as pd
    p = pd.read_parquet(bench / "pubmedqa_labeled.parquet")
    rows = [r for r in p.itertuples() if str(r.pubid) not in flags]
    rows = stratified([{"r": r} for r in rows], lambda x: x["r"].final_decision, n, rng)
    return [item(str(x["r"].pubid), "pubmedqa", "rag", x["r"].question, "yes/no/maybe", x["r"].final_decision,
                 "exact label; question-only setting by default (meta.context gives the reasoning-required setting)",
                 pmid=str(x["r"].pubid), context="\n".join(x["r"].context["contexts"])) for x in rows]


def matscibench(bench: Path, flags: set, rng: random.Random, n: int) -> list[dict]:
    import pandas as pd
    m = pd.read_parquet(bench / "matscibench.parquet")
    rows = [r for r in m.itertuples() if r.type == "NUM" and r.number_of_answers == "single"
            and len(r.image) == 0 and r.qid not in flags]
    rows = stratified([{"r": r} for r in rows], lambda x: x["r"].primary_category, n, rng)
    return [item(x["r"].qid, "matscibench", "rag", x["r"].question, "numeric", x["r"].answer,
                 "relative error <= 5% after unit check", unit=x["r"].unit, difficulty=x["r"].difficulty_level,
                 category=x["r"].primary_category) for x in rows]


def scicode(bench: Path) -> list[dict]:
    out = []
    for fname, sp in (("scicode_dev.jsonl", "dev"), ("scicode_test.jsonl", "test")):
        for x in map(json.loads, open(bench / fname)):
            it = item(x["problem_id"], "scicode", "code", x["problem_description_main"], "code",
                      None, "official SciCode unit tests per sub-step (needs the SciCode h5 test data)",
                      io=x["problem_io"], dependencies=x["required_dependencies"],
                      sub_steps=[{k: s[k] for k in ("step_number", "step_description_prompt", "function_header", "return_line")}
                                 for s in x["sub_steps"]])
            it["split"] = sp
            out.append(it)
    return out


def biocoder(bench: Path, flags: set) -> list[dict]:
    data = [x for x in json.loads((bench / "biocoder.json").read_text())
            if x["language"] == "Python" and x["test_case_id"] not in flags]
    return [item(x["test_case_id"], "biocoder", "code", x["promptSummaryOnly"], "code", None,
                 "official BioCoder test harness (Docker); goldenCode kept for reference",
                 signature=x["signature"], repository=x["repository"], file=x["filePath"],
                 golden_code=x["goldenCode"]) for x in data]


def scholar_claims(flags: set) -> list[dict]:
    data = [x for x in json.loads((ROOT / "journal_fixes/data/benchmarks/labbench_LitQA2.json").read_text())
            if x["id"] not in flags and x.get("key-passage") and x.get("sources")]
    return [item(f'claim-{x["id"]}', "scholar_claimpaper", "scholar",
                 f'Find the published paper that reports the following finding and give its DOI: "{x["key-passage"]}"',
                 "doi", [s.lower().replace("https://doi.org/", "") for s in x["sources"]],
                 "DOI exact match (case-insensitive) with any listed source", canary=x.get("canary")) for x in data]


# ── nf-core derived sets ─────────────────────────────────────────────────────

ACCELERATOR_FAMILY = re.compile(r"^(?:p\d|g\d|gr\d|inf\d|trn\d|dl\d|f\d|vt\d)")


def ec2_catalog() -> list[dict]:
    rows = []
    for r in csv.DictReader(open(ROOT / "journal_fixes/data/aws_ec2_us-east-1.csv")):
        # Burstable t-family instances cannot sustain pipeline workloads, so they are not valid answers.
        # Accelerator families (FPGA, Inferentia, Trainium, Gaudi, video) are not GPU-flagged in the catalog.
        if (r["current_gen"] != "Yes" or r["gpu"] or "Graviton" in r["processor"] or r["family"].startswith("t")
                or ACCELERATOR_FAMILY.match(r["family"])):
            continue
        try:
            rows.append({"instance": r["instance"], "vcpu": int(r["vcpu"]),
                         "mem": float(r["memory"].split()[0].replace(",", "")), "usd": float(r["usd_per_hour"])})
        except ValueError:
            continue
    return rows


def fitting(cat: list[dict], cpu: int, mem: float) -> dict:
    ok = sorted((r for r in cat if r["vcpu"] >= cpu and r["mem"] >= mem), key=lambda r: r["usd"])
    best = ok[0]["usd"]
    return {"cheapest": ok[0]["instance"], "cheapest_usd_per_hour": best,
            "acceptable": [r["instance"] for r in ok if r["usd"] <= best * 1.25]}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resembles(exe: str, tool: str) -> bool:
    a, b = _key(exe), _key(re.sub(r"\d+$", "", tool))
    return bool(a and b) and (a.startswith(b) or b.startswith(a) or a[:5] == b[:5])


def command_of(main_nf: str, tool: str) -> list[str] | None:
    """Executable (and subcommand) of the first command in the script block that runs the tool itself.
    None when no command matches the tool's name; such modules are left out rather than given a wrong gold."""
    m = re.search(r"\n\s*script:.*?\"\"\"(.*?)\"\"\"", main_nf, re.S)
    if not m:
        return None
    lines = m.group(1).splitlines()
    for i, line in enumerate(lines):
        toks = re.sub(r"^(\$\{[^}]*\}\s*)+", "", line.strip()).split()  # e.g. "${pre_command}samtools sort"
        if not toks or toks[0].startswith(("#", "$", "{", "}", "(", ")", "-", "|", ">", "<")):
            continue
        if toks[0] in SHELL_SKIP or "=" in toks[0]:
            continue
        exe = toks[0].split("/")[-1]
        if not re.match(r"^[A-Za-z][\w.+-]*$", exe) or not resembles(exe, tool):
            continue
        cont = [t for t in (toks[1:] + " ".join(l.strip() for l in lines[i + 1:i + 6]).split()) if t != "\\\\"]
        if exe in CAMEL_SUBCOMMAND:
            sub = next((t for t in cont if re.match(r"^[A-Z][A-Za-z]+$", t)), None)
        else:
            sub = cont[0] if cont and re.match(r"^[a-z][\w-]*$", cont[0]) else None
        return [exe, sub] if sub else [exe]
    return None


def nfcore_modules(nf_dir: Path) -> list[dict]:
    mods = []
    for main in sorted(glob.glob(str(nf_dir / "modules/nf-core/**/main.nf"), recursive=True)):
        d = Path(main).parent
        text = Path(main).read_text(errors="ignore")
        labels = set(re.findall(r'label\s+[\'"](process_\w+)[\'"]', text))
        if "process_gpu" in labels:
            continue
        lab = "process_high_memory" if "process_high_memory" in labels else next(
            (l for l in ("process_high", "process_medium", "process_low", "process_single") if l in labels), None)
        if not lab or not (d / "meta.yml").exists():
            continue
        try:
            meta = yaml.safe_load((d / "meta.yml").read_text())
        except yaml.YAMLError:
            continue
        tools = meta.get("tools") or []
        if not tools or not isinstance(tools[0], dict):
            continue
        tname, tinfo = next(iter(tools[0].items()))
        dois = [d.lower().rstrip(".") for d in DOI_RE.findall(str((tinfo or {}).get("doi") or ""))]
        cmd = command_of(text, tname)
        module = str(d.relative_to(nf_dir / "modules/nf-core"))
        if not dois or not cmd or cmd[0] in INTERPRETERS or module.startswith("custom/"):
            continue
        mods.append({"module": module, "tool": tname, "biotools": str((tinfo or {}).get("identifier") or ""),
                     "tool_description": " ".join(str((tinfo or {}).get("description") or "").split()),
                     "step_description": " ".join(str(meta.get("description") or "").split()),
                     "dois": dois, "homepage": (tinfo or {}).get("homepage"), "label": lab, "command": cmd})
    return mods


def add_biotools_dois(mods: list[dict]) -> None:
    """Add every publication DOI that the tool's bio.tools entry lists (public registry, cached)."""
    import urllib.request
    cache = json.loads(BIOTOOLS_CACHE.read_text()) if BIOTOOLS_CACHE.exists() else {}
    for m in mods:
        bt = m["biotools"].replace("biotools:", "").strip()
        if not bt:
            continue
        if bt not in cache:
            try:
                with urllib.request.urlopen(f"https://bio.tools/api/tool/{bt}?format=json", timeout=20) as r:
                    cache[bt] = [p["doi"].lower() for p in json.load(r).get("publication", []) if p.get("doi")]
            except Exception:
                cache[bt] = []
        m["dois"] += [x for x in cache[bt] if x not in m["dois"]]
    BIOTOOLS_CACHE.write_text(json.dumps(cache))


def nfcore_sets(nf_dir: Path, rng: random.Random, n: int) -> tuple[list, list, list, dict]:
    mods = nfcore_modules(nf_dir)
    add_biotools_dois(mods)
    cat = ec2_catalog()
    cloud_kb = "\n".join(p.read_text(errors="ignore").lower() for p in (ROOT / "cloud_agent/knowledge_base").glob("*.md"))
    rag_dois = {str(m.get("doi") or "").lower() for m in
                json.loads((ROOT / "rag_agent/knowledge_base/oa_corpus/manifest.json").read_text()).values()}
    by_tool = collections.defaultdict(list)
    for m in mods:
        by_tool[m["tool"]].append(m)
    tools = sorted(by_tool)
    rng.shuffle(tools)
    third = len(tools) // 3
    groups = {"cloud": tools[:third], "scholar": tools[third:2 * third], "full": tools[2 * third:]}

    def pick(tool_list, one_per_tool=True):
        return [rng.choice(by_tool[t]) if one_per_tool else by_tool[t][0] for t in tool_list]

    def flags(m):
        named = re.search(rf"(?<![\w-]){re.escape(m['tool'].lower())}(?![\w-])", cloud_kb) is not None
        return {"tool_named_in_cloud_kb": named, "tool_paper_in_rag_kb": any(x in rag_dois for x in m["dois"])}

    def resources(m):
        cpu, mem = NFCORE_LABELS[m["label"]]
        return {"label": m["label"], "cpus": cpu, "memory_gb": mem, **fitting(cat, cpu, mem)}

    cloud = []
    for m in stratified(pick(groups["cloud"]), lambda m: m["label"], n, rng):
        cloud.append(item(f'cloud-{m["module"]}', "cloud_nfcore", "cloud",
                          f'Pipeline step: {m["step_description"].rstrip(".")} (tool: {m["tool"]}). '
                          f'How many CPUs and how much memory should this step request by default, and which AWS EC2 '
                          f'on-demand instance (us-east-1, x86, current generation, no GPU) is the cheapest that fits it?',
                          "structured", resources(m),
                          "CPU and memory exact (nf-core default for the module's label); instance must fit and cost "
                          "<= 1.25x the cheapest fitting instance", module=m["module"], tool=m["tool"], **flags(m)))
    scholar = []
    for m in rng.sample(pick(groups["scholar"]), min(n, len(groups["scholar"]))):
        scholar.append(item(f'toolpaper-{m["module"]}', "scholar_toolpaper", "scholar",
                            f'Which publication introduced the tool {m["tool"]} ({m["tool_description"].rstrip(".")})? Give its DOI.',
                            "doi", m["dois"], "DOI exact match (case-insensitive) with any publication listed by nf-core or bio.tools",
                            module=m["module"], tool=m["tool"], **flags(m)))
    full = []
    for m in stratified(pick(groups["full"]), lambda m: m["label"], n, rng):
        full.append(item(f'full-{m["module"]}', "full_system_nfcore", "full_system",
                         f'Pipeline step: {m["step_description"].rstrip(".")} (tool: {m["tool"]}). '
                         f'(1) Which publication introduced {m["tool"]}? Give its DOI. '
                         f'(2) Give the command line that performs this step. '
                         f'(3) How many CPUs and how much memory should the job request, and which AWS EC2 on-demand '
                         f'instance (us-east-1, x86, current generation, no GPU) is the cheapest that fits it?',
                         "structured", {"literature": {"dois": m["dois"]},
                                        "code": {"command": m["command"]},
                                        "cloud": resources(m)},
                         "per part: (1) DOI exact match with any listed publication; (2) answer contains the executable and, if any, the subcommand; "
                         "(3) CPU and memory exact, instance fits and costs <= 1.25x the cheapest. Item passes if all three pass",
                         module=m["module"], tool=m["tool"], agents_needed=["scholar", "rag", "code", "cloud"], **flags(m)))
    stats = {"modules_parsed": len(mods), "tools": len(tools), "tools_per_group": {k: len(v) for k, v in groups.items()}}
    return cloud, scholar, full, stats


def main(bench: Path, nf_dir: Path, n: int) -> None:
    rng = random.Random(SEED)
    lf = {k: set(v) for k, v in json.loads((ROOT / "journal_fixes/data/leakage_flags.json").read_text()).items()}
    per_agent = {
        "rag_litqa2": labbench("LitQA2", "rag", lf["labbench_LitQA2"], rng),
        "rag_pubmedqa": pubmedqa(bench, lf["pubmedqa"], rng, n),
        "rag_matscibench": matscibench(bench, lf["matscibench"], rng, n),
        "rag_protocolqa": labbench("ProtocolQA", "rag", lf["labbench_ProtocolQA"], rng),
        "code_scicode": scicode(bench),
        "code_biocoder": biocoder(bench, lf["biocoder"]),
        "code_seqqa": labbench("SeqQA", "code", lf["labbench_SeqQA"], rng, n),
        "scholar_claimpaper": scholar_claims(lf["labbench_LitQA2"]),
    }
    cloud, scholar, full, nf_stats = nfcore_sets(nf_dir, rng, n)
    per_agent["cloud_nfcore"], per_agent["scholar_toolpaper"] = cloud, scholar

    (OUT / "per_agent").mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "dev_fraction": DEV_FRACTION, "nfcore": nf_stats, "per_agent": {}, "full_system": {}}
    for name, items in per_agent.items():
        if name != "code_scicode":
            items = split(items, rng)
        with open(OUT / "per_agent" / f"{name}.jsonl", "w") as f:
            for x in items:
                f.write(json.dumps(x) + "\n")
        summary["per_agent"][name] = {"agent": items[0]["agent"], "n": len(items),
                                      "dev": sum(x["split"] == "dev" for x in items),
                                      "test": sum(x["split"] == "test" for x in items)}
    full = split(full, rng)
    with open(OUT / "full_system_nfcore.jsonl", "w") as f:
        for x in full:
            f.write(json.dumps(x) + "\n")
    summary["full_system"] = {"n": len(full), "dev": sum(x["split"] == "dev" for x in full),
                              "test": sum(x["split"] == "test" for x in full),
                              "labels": dict(collections.Counter(x["gold"]["cloud"]["label"] for x in full)),
                              "tool_named_in_cloud_kb": sum(x["meta"]["tool_named_in_cloud_kb"] for x in full),
                              "tool_paper_in_rag_kb": sum(x["meta"]["tool_paper_in_rag_kb"] for x in full)}
    summary["cloud_label_mix"] = dict(collections.Counter(x["gold"]["label"] for x in cloud))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=str(Path.home() / "otr_local" / "benchmarks"))
    ap.add_argument("--nfcore-dir", default=str(Path.home() / "otr_local" / "nfcore_modules"))
    ap.add_argument("--n", type=int, default=150, help="sample size for the sampled sets")
    a = ap.parse_args()
    main(Path(a.bench_dir), Path(a.nfcore_dir), a.n)
