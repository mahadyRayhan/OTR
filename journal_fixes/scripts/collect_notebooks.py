"""
collect_notebooks.py — real, maintained tutorial notebooks for the Notebook (code) agent.

Pulls .ipynb files from openly licensed project repositories through the GitHub
API (no clone, no execution), strips outputs, and writes them next to the
existing notebooks so code_agent/build_vector_db.py picks them up.

Output
  code_agent/knowledge_base/gh__<owner>__<repo>__<path>.ipynb
  code_agent/knowledge_base/notebook_manifest.json  (source URL, licence, commit)

Usage
  python journal_fixes/scripts/collect_notebooks.py --per-repo 40
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "code_agent" / "knowledge_base"

# (owner, repo, domain). All are public tutorial/documentation repositories.
REPOS = [
    ("scverse", "scanpy-tutorials", "bioinformatics"),
    ("scverse", "scvi-tutorials", "bioinformatics"),
    ("scverse", "squidpy_notebooks", "bioinformatics"),
    ("theislab", "single-cell-best-practices", "bioinformatics"),
    ("saezlab", "decoupler-py", "bioinformatics"),
    ("materialsproject", "workshop", "materials_science"),
    ("Project-MONAI", "tutorials", "health_science"),
]

s = requests.Session()
s.headers["Accept"] = "application/vnd.github+json"


def repo_info(owner: str, repo: str) -> dict:
    r = s.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=30)
    r.raise_for_status()
    j = r.json()
    return {"branch": j["default_branch"], "license": (j.get("license") or {}).get("spdx_id")}


def list_notebooks(owner: str, repo: str, branch: str) -> tuple[str, list[str]]:
    r = s.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", timeout=60)
    r.raise_for_status()
    j = r.json()
    paths = [t["path"] for t in j["tree"] if t["type"] == "blob" and t["path"].endswith(".ipynb")
             and not re.search(r"(^|/)(\.[^/]*|_build|build|archive|deprecated|old)/", t["path"])]
    return j["sha"], sorted(paths)


def strip(nb: dict) -> dict:
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-repo", type=int, default=40)
    args = ap.parse_args()
    manifest_path = OUT_DIR / "notebook_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for owner, repo, domain in REPOS:
        try:
            info = repo_info(owner, repo)
            sha, paths = list_notebooks(owner, repo, info["branch"])
        except Exception as exc:
            print(f"[skip] {owner}/{repo}: {exc}")
            continue
        kept = 0
        for path in paths:
            if kept >= args.per_repo:
                break
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}"
            try:
                r = requests.get(raw, timeout=60)
                r.raise_for_status()
                nb = strip(r.json())
            except Exception:
                continue
            n_code = sum(1 for c in nb.get("cells", []) if c.get("cell_type") == "code" and "".join(c.get("source", [])).strip())
            if n_code < 3:
                continue
            name = "gh__" + "__".join([owner, repo, path.replace("/", "__")])
            (OUT_DIR / name).write_text(json.dumps(nb), encoding="utf-8")
            manifest[name] = {
                "repo": f"{owner}/{repo}", "path": path, "commit": sha, "license": info["license"],
                "domain": domain, "url": f"https://github.com/{owner}/{repo}/blob/{sha}/{path}",
                "code_cells": n_code,
            }
            kept += 1
        print(f"{owner}/{repo:28s} {kept:3d} notebooks  licence={info['license']}")
        manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"Total notebooks: {len(manifest)}")


if __name__ == "__main__":
    main()
