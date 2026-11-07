"""
refine_paper_gold.py — make the gold DOIs of the paper-lookup sets name publications, by rule (no LLM).

  1. A bioRxiv preprint DOI (10.1101/...) also accepts the DOI of its published version, from the
     bioRxiv API (api.biorxiv.org/pubs), else Crossref's "is-preprint-of" relation, else Europe PMC's
     "Preprint of" link. The question asks for the paper; either version is it.
     Conversely, a published DOI also accepts its preprint versions (bioRxiv/medRxiv API, Crossref
     "has-preprint"): the same paper, so citing the preprint is not wrong.
  2. Items whose gold DOIs are all software archives (Zenodo, figshare) are dropped: those DOIs come
     from nf-core meta.yml and name a code deposit, not the publication the question asks for.

Applies to every item of the domain's scholar_toolpaper, scholar_claimpaper and full_system sets,
dev and test alike; ids and splits are unchanged. Records what changed in <set>.refine_log.json.

Usage: python journal_fixes/scripts/refine_paper_gold.py --domain bio
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path.home() / "otr_local" / "biorxiv_published_cache.json"
SOFTWARE = ("10.5281/zenodo", "10.6084/m9.figshare")


def _crossref_published(doi: str, cache: dict) -> str | None:
    key = "crossref:" + doi
    if key not in cache:
        try:
            with urllib.request.urlopen(f"https://api.crossref.org/works/{doi}", timeout=30) as r:
                rel = json.load(r)["message"].get("relation", {}).get("is-preprint-of", [])
            cache[key] = next((x["id"].lower() for x in rel if x.get("id-type") == "doi"), None)
        except Exception as e:
            print("  Crossref API error", doi, e)
            return None
        time.sleep(1)
    return cache[key]


def _epmc_published(doi: str, cache: dict) -> str | None:
    key = "europepmc:" + doi
    if key not in cache:
        base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=core&query="
        try:
            with urllib.request.urlopen(base + urllib.parse.quote(f'DOI:"{doi}"'), timeout=30) as r:
                hits = json.load(r)["resultList"]["result"]
            links = [c for h in hits for c in (h.get("commentCorrectionList") or {}).get("commentCorrection", [])
                     if c.get("type") == "Preprint of"]
            pub = None
            if links:
                q = f'EXT_ID:{links[0]["id"]} AND SRC:{links[0].get("source", "MED")}'
                with urllib.request.urlopen(base + urllib.parse.quote(q), timeout=30) as r:
                    res = json.load(r)["resultList"]["result"]
                pub = (res[0].get("doi") or "").lower() or None if res else None
            cache[key] = pub
        except Exception as e:
            print("  Europe PMC API error", doi, e)
            return None
        time.sleep(1)
    return cache[key]


def published(doi: str, cache: dict) -> str | None:
    doi = re.sub(r"v\d+$", "", doi)  # nf-core sometimes records the version suffix
    if doi not in cache:
        try:
            with urllib.request.urlopen(f"https://api.biorxiv.org/pubs/biorxiv/{doi}", timeout=30) as r:
                coll = json.load(r).get("collection") or []
            cache[doi] = (coll[0].get("published_doi") or "").lower() or None if coll else None
        except Exception as e:
            print("  bioRxiv API error", doi, e)
            return None  # not cached, so a rerun retries it
        time.sleep(1)
    return cache[doi] or _crossref_published(doi, cache) or _epmc_published(doi, cache)


def preprints(doi: str, cache: dict) -> list[str]:
    """Preprint versions of a published DOI: bioRxiv/medRxiv API, plus Crossref's "has-preprint" relation."""
    key = "preprints:" + doi
    if key not in cache:
        found = []
        try:
            for server in ("biorxiv", "medrxiv"):
                with urllib.request.urlopen(f"https://api.biorxiv.org/pubs/{server}/{doi}", timeout=30) as r:
                    found += [(c.get("preprint_doi") or "").lower() for c in json.load(r).get("collection") or []]
            with urllib.request.urlopen(f"https://api.crossref.org/works/{doi}", timeout=30) as r:
                rel = json.load(r)["message"].get("relation", {}).get("has-preprint", [])
            found += [x["id"].lower() for x in rel if x.get("id-type") == "doi"]
        except Exception as e:
            print("  preprint lookup error", doi, e)
            return []  # not cached, so a rerun retries it
        cache[key] = sorted({d for d in found if d})
        time.sleep(1)
    return cache[key]


def gold_list(x: dict) -> list[str]:
    return x["gold"]["literature"]["dois"] if isinstance(x["gold"], dict) else x["gold"]


def main(domain: str) -> None:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for name in ("scholar_toolpaper", "scholar_claimpaper", "full_system"):
        f = ROOT / "journal_fixes/data/testsets" / domain / f"{name}.jsonl"
        rows = [json.loads(l) for l in open(f)]
        log = {"added_published_versions": {}, "added_preprint_versions": {}, "dropped_software_only": []}
        keep = []
        for x in rows:
            dois = gold_list(x)
            if dois and all(d.lower().startswith(SOFTWARE) for d in dois):
                log["dropped_software_only"].append({"id": x["id"], "split": x.get("split"), "gold": dois})
                continue
            for d in list(dois):
                if d.lower().startswith("10.1101/"):
                    pub = published(d.lower(), cache)
                    if pub and pub not in (g.lower() for g in dois):
                        dois.append(pub)
                        log["added_published_versions"].setdefault(x["id"], []).append({d: pub})
                elif not d.lower().startswith(SOFTWARE):
                    for pre in preprints(d.lower(), cache):
                        if pre not in (g.lower() for g in dois):
                            dois.append(pre)
                            log["added_preprint_versions"].setdefault(x["id"], []).append({d: pre})
            keep.append(x)
        CACHE.write_text(json.dumps(cache))
        f.write_text("".join(json.dumps(x) + "\n" for x in keep))
        log_file = f.parent / f"{name}.refine_log.json"  # cumulative over reruns
        old = json.loads(log_file.read_text()) if log_file.exists() else {}
        for kind in ("added_published_versions", "added_preprint_versions"):
            for k, v in old.get(kind, {}).items():
                log[kind].setdefault(k, v)
        log["dropped_software_only"] = old.get("dropped_software_only", []) + log["dropped_software_only"]
        log_file.write_text(json.dumps(log, indent=1))
        print(f"{name}: {len(rows)} -> {len(keep)} items; published versions added for "
              f"{len(log['added_published_versions'])} items, preprint versions for "
              f"{len(log['added_preprint_versions'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="bio")
    main(ap.parse_args().domain)
