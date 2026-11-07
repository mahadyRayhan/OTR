"""
collect_aws_catalog.py — real EC2 instance specs and on-demand prices for the Cloud agent.

Source: the official AWS Price List bulk API (EC2, us-east-1, CSV).
Keeps on-demand, Linux, shared-tenancy, no pre-installed software rows, and writes:

  cloud_agent/knowledge_base/aws_ec2_catalog_<category>.md  (one table per instance family)
  journal_fixes/data/aws_ec2_us-east-1.csv                  (the filtered rows, for analysis)

Usage
  python journal_fixes/scripts/collect_aws_catalog.py
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
KB_DIR = ROOT / "cloud_agent" / "knowledge_base"
DATA_DIR = ROOT / "journal_fixes" / "data"
BASE = "https://pricing.us-east-1.amazonaws.com"

CATEGORY = {  # first letters of the family -> AWS category
    "General purpose": ("m", "t", "mac", "a"),
    "Compute optimized": ("c",),
    "Memory optimized": ("r", "x", "u", "z"),
    "Accelerated computing": ("p", "g", "inf", "trn", "dl", "f", "vt"),
    "Storage optimized": ("i", "d", "h", "im", "is"),
    "HPC optimized": ("hpc",),
}


def category_of(family: str) -> str:
    best = ("Other", 0)
    for cat, prefixes in CATEGORY.items():
        for p in prefixes:
            if family.startswith(p) and len(p) > best[1]:
                best = (cat, len(p))
    return best[0]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    idx = requests.get(f"{BASE}/offers/v1.0/aws/AmazonEC2/current/region_index.json", timeout=60).json()
    url = BASE + idx["regions"]["us-east-1"]["currentVersionUrl"].replace("index.json", "index.csv")
    version = url.split("/")[-3]
    # The raw offer file is ~300 MB; keep it out of the synced project folder.
    raw = Path(os.environ.get("AWS_RAW_DIR", "/tmp")) / f"aws_ec2_us-east-1_{version}.csv"
    if not raw.exists():
        subprocess.run(["curl", "-sSL", "-o", str(raw), url], check=True)

    rows = []
    with open(raw, newline="", encoding="utf-8") as f:
        for _ in range(5):  # metadata lines before the header
            next(f)
        for r in csv.DictReader(f):
            if (r.get("TermType") == "OnDemand" and r.get("Product Family") == "Compute Instance"
                    and r.get("Operating System") == "Linux" and r.get("Tenancy") == "Shared"
                    and r.get("Pre Installed S/W") == "NA" and r.get("CapacityStatus") == "Used"
                    and r.get("Unit") == "Hrs"):
                try:
                    price = float(r["PricePerUnit"])
                except ValueError:
                    continue
                if price <= 0:
                    continue
                rows.append({
                    "instance": r["Instance Type"], "family": r["Instance Type"].split(".")[0],
                    "vcpu": r["vCPU"], "memory": r["Memory"], "gpu": r.get("GPU") or "",
                    "gpu_memory": r.get("GPU Memory") or "", "storage": r["Storage"],
                    "network": r["Network Performance"], "processor": r["Physical Processor"],
                    "current_gen": r["Current Generation"], "usd_per_hour": price,
                })

    rows.sort(key=lambda x: (x["family"], x["usd_per_hour"]))
    with open(DATA_DIR / "aws_ec2_us-east-1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by_cat: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["current_gen"] != "Yes":
            continue
        by_cat[category_of(r["family"])][r["family"]].append(r)

    for cat, fams in by_cat.items():
        slug = re.sub(r"[^a-z]+", "_", cat.lower()).strip("_")
        lines = [
            f"# AWS EC2 catalog: {cat} instances (us-east-1, on-demand Linux)",
            "",
            f"Source: AWS Price List bulk API, EC2 offer version {version}, retrieved {date.today().isoformat()}.",
            "Prices are on-demand USD per hour, shared tenancy, no pre-installed software. Spot and savings-plan prices are lower.",
            "",
        ]
        for fam in sorted(fams):
            items = fams[fam]
            lines += [f"## {fam} family ({cat})", "",
                      f"Processor: {items[0]['processor']}. Network: up to {items[-1]['network']}.", "",
                      "| Instance | vCPU | Memory | GPU | GPU memory | Instance storage | USD/hour |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in items:
                lines.append(f"| {r['instance']} | {r['vcpu']} | {r['memory']} | {r['gpu'] or '-'} | "
                             f"{r['gpu_memory'] or '-'} | {r['storage']} | {r['usd_per_hour']:.4f} |")
            lines.append("")
        (KB_DIR / f"aws_ec2_catalog_{slug}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"{cat:24s} {len(fams):3d} families")
    print(f"Rows kept: {len(rows)}; offer version {version}")


if __name__ == "__main__":
    main()
