"""
collect_profiler_queries.py — real researcher questions for training and testing the
User Preference Profiler (replaces the 32 author-written training phrases).

Source: Stack Exchange API (Bioinformatics and Matter Modeling sites; CC BY-SA).
The questions are unlabeled here; label_profiler_queries.py adds PAC labels.

Output: journal_fixes/data/profiler_queries.json

Usage
  python journal_fixes/scripts/collect_profiler_queries.py --pages 4
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "journal_fixes" / "data" / "profiler_queries.json"
SITES = {"bioinformatics": "bioinformatics", "mattermodeling": "materials_science"}


def clean(body: str) -> str:
    body = re.sub(r"<pre.*?</pre>", " [code] ", body, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", html.unescape(body)).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=4)
    args = ap.parse_args()
    rows = {}
    for site, domain in SITES.items():
        for page in range(1, args.pages + 1):
            r = requests.get("https://api.stackexchange.com/2.3/questions", params={
                "site": site, "sort": "votes", "order": "desc", "pagesize": 100, "page": page, "filter": "withbody"}, timeout=60)
            r.raise_for_status()
            j = r.json()
            for q in j.get("items", []):
                text = f"{html.unescape(q['title'])}. {clean(q['body'])}"
                rows[f"{site}:{q['question_id']}"] = {
                    "id": f"{site}:{q['question_id']}", "site": site, "domain": domain,
                    "title": html.unescape(q["title"]), "text": text[:1500], "tags": q.get("tags", []),
                    "score": q.get("score"), "url": q.get("link"), "license": "CC BY-SA (Stack Exchange)"}
            print(f"{site} page {page}: {len(j.get('items', []))} (quota left {j.get('quota_remaining')})")
            if not j.get("has_more"):
                break
            time.sleep(j.get("backoff", 0) + 0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(list(rows.values()), indent=1))
    print(f"Saved {len(rows)} questions to {OUT}")


if __name__ == "__main__":
    main()
