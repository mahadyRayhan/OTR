"""
biocoder_eval.py — score code_biocoder answers by BioCoder's fuzz test, in a Docker container.

BioCoder (Tang et al., 2024) gives each item a context file: the repository code around the target
function, a <<insert solution here>> slot, and a main() whose inputs are placeholders such as
<|int;range=0,100|> or <|string|>. An answer passes when, for every filled-in variant, the context with
the answer prints the same output as the context with the golden function. This is a re-implementation
of that test (same context files, placeholders and output comparison), not the authors' harness.

  extract(answer, name)   the imports, functions and classes of the answer's code block that defines name
  fill(context, rng)      the context with its placeholders replaced by random values
  check(item, answer)     {"correct", "answered", "parsed", "parts"} like scoring.score(); TRIALS variants;
                          correct is None when the golden code fails (2 test items, cnvkit, a NaN input)

The model's code runs only here, for evaluation, inside the container: no network, read-only file
system except a small /tmp, 1 CPU, 2 GB memory, a time limit per run. SciWeaver itself runs no code.

Setup:     docker build -t sciweaver-biocoder:1 journal_fixes/biocoder_docker
Drop items whose prompt names a function the test never calls (8 of 86):
           python journal_fixes/scripts/biocoder_eval.py --drop-misnamed bio
Self-test: python journal_fixes/scripts/biocoder_eval.py --selftest [--split dev]
           (golden code must pass every item it is run on, and a wrong function must fail)
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import string
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path.home() / "otr_local" / "benchmarks" / "biocoder.json"
IMAGE = "sciweaver-biocoder:1"
TRIALS = 5
RUN_SECONDS = 30
SLOT = re.compile(r"^([ \t]*)<<insert solution here>>[ \t]*$", re.M)
PLACEHOLDER = re.compile(r"<\|(\w+)((?:;[^|]*)?)\|>")
CODE_BLOCK = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.S | re.I)
KEEP = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

RUNNER = r'''
import json, os, shutil, subprocess, sys, tempfile
out = {}
for f in sorted(os.listdir("/w")):
    if not f.endswith(".py"):
        continue
    with tempfile.TemporaryDirectory(dir="/tmp") as d:  # a writable copy: some contexts write beside __file__
        shutil.copy("/w/" + f, d)
        try:
            p = subprocess.run([sys.executable, f], cwd=d, capture_output=True, text=True, timeout=%d)
            out[f] = {"rc": p.returncode, "stdout": p.stdout[-20000:], "stderr": p.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            out[f] = {"rc": "timeout", "stdout": "", "stderr": ""}
print(json.dumps(out))
''' % RUN_SECONDS

_SOURCE = None


def source(item_id: str) -> dict:
    global _SOURCE
    if _SOURCE is None:
        _SOURCE = {x["test_case_id"]: x for x in json.loads(BENCH.read_text())}
    return _SOURCE[item_id]


def function_name(signature: str) -> str:
    return re.search(r"^\s*(?:async\s+)?def\s+(\w+)", signature, re.M).group(1)  # may follow a decorator


def extract(answer: str, name: str) -> str | None:
    """Imports and definitions from the answer's last code block that defines `name` (no example calls)."""
    blocks = CODE_BLOCK.findall(answer or "") or [answer or ""]
    for block in reversed(blocks):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree)):
            return "\n\n".join(ast.unparse(n) for n in tree.body if isinstance(n, KEEP))
    return None


def _value(kind: str, opts: str, rng: random.Random) -> str:
    o = dict(kv.split("=", 1) for kv in opts.strip(";").split(";") if "=" in kv)
    if kind == "int":
        lo, hi = map(int, o.get("range", "0,100").split(","))
        return str(rng.randint(lo, hi))
    if kind == "float":
        lo, hi = map(float, o.get("range", "0,1").split(","))
        return repr(rng.uniform(lo, hi))
    if kind == "bool":
        return rng.choice(["True", "False"])
    if kind == "string":
        return repr("".join(rng.choice(string.ascii_letters) for _ in range(int(o.get("length", 10)))))
    raise ValueError(f"unknown placeholder <|{kind}{opts}|>")


def fill(context: str, rng: random.Random) -> str:
    return PLACEHOLDER.sub(lambda m: _value(m.group(1), m.group(2), rng), context)


def _program(context: str, code: str) -> str:
    m = SLOT.search(context)
    indented = "\n".join(m.group(1) + line if line.strip() else line for line in code.splitlines())
    return context[:m.start()] + indented + context[m.end():]


def _run(files: dict[str, str]) -> dict:
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            Path(d, name).write_text(text)
        cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
               "--memory", "2g", "--cpus", "1", "--pids-limit", "256", "-v", f"{d}:/w:ro", IMAGE,
               "python", "-c", RUNNER]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_SECONDS * len(files) + 120)
        if p.returncode != 0:
            raise RuntimeError(f"docker run failed: {p.stderr[-500:]}")
        return json.loads(p.stdout)


def check(item: dict, answer: str) -> dict:
    src = source(item["id"])
    code = extract(answer, function_name(src["signature"]))
    if code is None:
        return {"correct": False, "answered": False, "parsed": None, "parts": {}}
    files = {}
    for t in range(TRIALS):
        ctx = fill(src["contextCode"], random.Random(t))
        files[f"golden_{t}.py"] = _program(ctx, src["goldenCode"])
        files[f"answer_{t}.py"] = _program(ctx, code)
    res = _run(files)
    golden_ok = all(res[f"golden_{t}.py"]["rc"] == 0 for t in range(TRIALS))
    same = [res[f"answer_{t}.py"]["rc"] == 0 and res[f"answer_{t}.py"]["stdout"] == res[f"golden_{t}.py"]["stdout"]
            for t in range(TRIALS)]
    first_fail = next((res[f"answer_{t}.py"] for t in range(TRIALS) if not same[t]), None)
    # The golden code itself fails on 2 of the 86 items (test split) in this environment: unscorable, not wrong.
    return {"correct": all(same) if golden_ok else None, "answered": True, "parsed": code,
            "parts": {"trials_passed": sum(same), "trials": TRIALS, "golden_ok": golden_ok,
                      "first_failure": first_fail and (first_fail["stderr"][-300:] or "output differs")}}


def misnamed(item: dict) -> bool:
    """The prompt asks for a function by a name that BioCoder's test never calls (e.g. get_candidates vs candidates)."""
    asked = re.findall(r"\b(?:called|named)\s+[\"'`]?(\w+)", item["question"])
    return bool(asked) and function_name(item["meta"]["signature"]) not in asked


def drop_misnamed(domain: str) -> None:
    """Remove misnamed items from the set (ids and splits of the rest unchanged); log them."""
    f = ROOT / "journal_fixes/data/testsets" / domain / "code_biocoder.jsonl"
    rows = [json.loads(l) for l in open(f)]
    drop = [x for x in rows if misnamed(x)]
    f.write_text("".join(json.dumps(x) + "\n" for x in rows if not misnamed(x)))
    log = f.parent / "code_biocoder.refine_log.json"
    old = json.loads(log.read_text()) if log.exists() else {"dropped_misnamed": []}
    old["dropped_misnamed"] += [{"id": x["id"], "split": x["split"], "test_calls": function_name(x["meta"]["signature"])}
                                for x in drop]
    log.write_text(json.dumps(old, indent=1))
    print(f"code_biocoder: {len(rows)} -> {len(rows) - len(drop)} items")


def selftest(split: str | None) -> None:
    items = [json.loads(l) for l in open(ROOT / "journal_fixes/data/testsets/bio/code_biocoder.jsonl")]
    items = [x for x in items if split is None or x["split"] == split]
    broken = []
    for x in items:
        src = source(x["id"])
        name = function_name(src["signature"])
        good = check(x, f"```python\n{src['goldenCode']}\n```")
        wrong = check(x, f"```python\ndef {name}(*args, **kwargs):\n    return 'wrong'\n```")
        print(f"{x['id'][:12]} {src['repository']:<24} golden {good['correct']!s:<5} wrong {wrong['correct']!s:<5}"
              + ("" if good["correct"] else f"  {good['parts']['first_failure']}"), flush=True)
        if good["correct"] is not True or wrong["correct"]:
            broken.append(x["id"])
    print(f"\n{len(items) - len(broken)}/{len(items)} items behave as expected; broken: {broken}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--split")
    ap.add_argument("--drop-misnamed", metavar="DOMAIN", help="remove items whose prompt names another function")
    a = ap.parse_args()
    if a.drop_misnamed:
        drop_misnamed(a.drop_misnamed)
    if a.selftest:
        selftest(a.split)
