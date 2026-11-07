"""
profiler_eval.py — validate the PAC intent classifier on real, held-out researcher questions.

Steps
  label      Two different LLM labellers (Config.EVAL_JUDGES) label each
             Stack Exchange question as Procedural / Analytical / Conceptual.
             Cohen's kappa between them is reported; only agreed labels are kept.
  sheet      Writes a CSV of 150 random questions for human annotation (the stronger fix:
             2 annotators, report kappa, use as the test set).
  evaluate   Stratified 70/30 split. Trains the paper's model (bag of words + Multinomial
             NB, alpha=1) and a TF-IDF logistic-regression comparison on the train split,
             and scores both plus the ORIGINAL 32-phrase model on the held-out split.
             If profiler_human_annotation.csv has labels, it is used as the test set.

Output: journal_fixes/results/profiler_eval.json

Usage
  python journal_fixes/scripts/profiler_eval.py sheet
  python journal_fixes/scripts/profiler_eval.py label
  python journal_fixes/scripts/profiler_eval.py evaluate
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

DATA = ROOT / "journal_fixes" / "data"
QUERIES = DATA / "profiler_queries.json"
LABELS = DATA / "profiler_llm_labels.json"
SHEET = DATA / "profiler_human_annotation.csv"
OUT = ROOT / "journal_fixes" / "results" / "profiler_eval.json"
CLASSES = ["procedural", "analytical", "conceptual"]
import config  # noqa: E402

LABELLERS = config.Config.EVAL_JUDGES  # two different models; neither is the system model

PROMPT = """Classify what this researcher mainly wants, using Bloom's revised taxonomy as in the definitions:
- procedural: code, commands, pipelines, step-by-step instructions to DO something
- analytical: comparisons, trade-offs, choosing between methods, evaluating or diagnosing results
- conceptual: definitions, mechanisms, theory, explanations of WHY or HOW something works
Reply with exactly one word: procedural, analytical or conceptual.

Question: {q}"""


async def label_with(model: str, q: str) -> str:
    from journal_fixes.scripts.llm import generate
    return (await generate(model, PROMPT.format(q=q))).strip().lower()


def norm(x: str) -> str | None:
    m = re.search(r"procedural|analytical|conceptual", x or "")
    return m.group(0) if m else None


async def do_label() -> None:
    qs = json.loads(QUERIES.read_text())
    done = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    sem = asyncio.Semaphore(8)

    async def one(q):
        if q["id"] in done:
            return
        async with sem:
            try:
                a, b = await asyncio.gather(*[label_with(m, q["text"]) for m in LABELLERS])
                done[q["id"]] = {"labeller_a": norm(a), "labeller_b": norm(b)}
            except Exception as exc:
                if type(exc).__name__ == "CreditsExhausted":
                    raise
                print("label error", q["id"], repr(exc)[:120])

    await asyncio.gather(*[one(q) for q in qs])
    LABELS.write_text(json.dumps(done, indent=1))
    print(f"labelled {len(done)} / {len(qs)}")


def do_sheet() -> None:
    qs = json.loads(QUERIES.read_text())
    random.Random(1).shuffle(qs)
    with SHEET.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "url", "text", "annotator_1", "annotator_2"])
        for q in qs[:150]:
            w.writerow([q["id"], q["url"], q["text"][:800], "", ""])
    print(f"wrote {SHEET}: fill annotator columns with procedural / analytical / conceptual")


def do_evaluate() -> None:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    qs = {q["id"]: q for q in json.loads(QUERIES.read_text())}
    lab = json.loads(LABELS.read_text())
    pairs = [(v["labeller_a"], v["labeller_b"]) for v in lab.values() if v["labeller_a"] and v["labeller_b"]]
    kappa = cohen_kappa_score([a for a, _ in pairs], [b for _, b in pairs])
    agreed = [(qs[k]["text"], v["labeller_a"]) for k, v in lab.items() if v["labeller_a"] and v["labeller_a"] == v["labeller_b"]]
    X, y = [t for t, _ in agreed], [l for _, l in agreed]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    test_source = "llm_agreed_heldout"

    human = []
    if SHEET.exists():
        for row in csv.DictReader(SHEET.open()):
            a, b = norm(row["annotator_1"]), norm(row["annotator_2"])
            if a and a == b:
                human.append((row["text"], a))
    if len(human) >= 30:
        Xte, yte, test_source = [t for t, _ in human], [l for _, l in human], "human_agreed"
        held = {t for t in Xte}
        Xtr, ytr = [t for t in X if t not in held], [l for t, l in zip(X, y) if t not in held]

    from profiler_agent.agent import _NB_TRAINING_DATA
    orig_X = [t for t, _ in _NB_TRAINING_DATA]
    orig_y = [d.value.lower() for _, d in _NB_TRAINING_DATA]

    def score(model, Xa, ya):
        model.fit(Xa, ya)
        p = model.predict(Xte)
        rng = np.random.default_rng(0)
        idx = [rng.integers(0, len(yte), len(yte)) for _ in range(2000)]
        boot = [accuracy_score(np.array(yte)[i], p[i]) for i in idx]
        return {"accuracy": round(accuracy_score(yte, p), 3), "macro_f1": round(f1_score(yte, p, average="macro"), 3),
                "ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]}

    nb = lambda: make_pipeline(CountVectorizer(stop_words="english"), MultinomialNB(alpha=1.0))
    majority = max(set(yte), key=yte.count)
    report = {
        "labellers": LABELLERS,
        "labeller_kappa": round(float(kappa), 3), "n_labelled": len(pairs), "n_agreed": len(agreed),
        "class_counts": {c: y.count(c) for c in CLASSES}, "test_source": test_source,
        "n_train": len(Xtr), "n_test": len(Xte),
        "majority_baseline_accuracy": round(yte.count(majority) / len(yte), 3),
        "original_32_phrase_model": score(nb(), orig_X, orig_y),
        "mnb_retrained_on_real_questions": score(nb(), Xtr, ytr),
        "tfidf_logreg_retrained": score(make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), LogisticRegression(max_iter=2000, C=4)), Xtr, ytr),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    {"label": lambda: asyncio.run(do_label()), "sheet": do_sheet, "evaluate": do_evaluate}[step]()
