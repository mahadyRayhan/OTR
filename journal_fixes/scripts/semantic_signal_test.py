"""
semantic_signal_test.py — can semantic consensus replace the LLM judge? Offline, local models only, no API calls.

Reads node_signals.json (written by node_signal_analysis.py: each checkable node, labelled by whether its
answer is the gold answer) and the run's traces, then scores each node with signals that need no gold answer
and no LLM:

  sim_full     mean cosine similarity of the node's whole output to the other nodes' outputs (MiniLM)
  sim_answer   the same, on the answer span only (the last part of the output; an MCQ letter is replaced by
               the text of the option it names, so two nodes choosing the same option map to the same meaning)
  nli_agree    mean probability that another node's answer span entails this one (DeBERTa NLI, both ways)
  nli_support  highest probability that one of the node's retrieved documents entails its answer span
  goal_rel     cosine similarity of the node's own sub-task and its answer span (does it answer its goal?)

Reported: AUROC of each signal against the gold label; the accuracy of picking each item's final answer by a
signal; and a logistic combination of the signals, fitted leave-one-item-out so no item scores itself.

Usage: python journal_fixes/scripts/semantic_signal_test.py [--run ~/otr_local/runs/pilot_bio_shared_small]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from journal_fixes.scripts.node_signal_analysis import LOCAL, Docs, auroc, load_items  # noqa: E402

SIGNALS = ["judge", "agreement", "sim_full", "sim_answer", "nli_agree", "nli_support", "goal_rel"]
SPAN = 600


def answer_span(item: dict, row: dict, text: str) -> str:
    cut = re.split(r"(?:ANSWER|RESULT)\s*[:：]", text)
    body = (cut[-2] if len(cut) > 1 else text)[-SPAN:]
    if row["kind"] == "mcq" and item.get("meta", {}).get("choices"):
        i = ord(row["value"]) - 65
        ch = item["meta"]["choices"]
        if i < len(ch):
            return f"The answer is: {ch[i]}"
    return body.strip()


def main(run: Path, domain: str) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")  # labels: contradiction, entailment, neutral
    ent = list(nli.model.config.label2id).index("entailment") if hasattr(nli.model.config, "label2id") else 1

    def entail(pairs):
        if not pairs:
            return np.array([])
        logits = np.asarray(nli.predict(pairs, batch_size=16, show_progress_bar=False))
        p = np.exp(logits - logits.max(1, keepdims=True))
        return (p / p.sum(1, keepdims=True))[:, ent]

    items, docs = load_items(domain), Docs(LOCAL / "indexes" / domain)
    rows = json.load(open(run / "node_signals.json"))["node_rows"]
    trace_of = {json.loads(l)["id"]: json.loads(l)["sciweaver"].get("trace_file") for l in open(run / "answers.jsonl")}
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item"]].append(r)

    for iid, rs in by_item.items():
        t = json.load(open(trace_of[iid]))
        item = items[iid]
        outs = [t["nodes"][r["node"]].get("output") or "" for r in rs]
        spans = [answer_span(item, r, o) for r, o in zip(rs, outs)]
        goals = [str(t["nodes"][r["node"]].get("task") or t["nodes"][r["node"]].get("agent_input") or "")[:1000] for r in rs]
        E_full = emb.encode(outs, normalize_embeddings=True)
        E_ans = emb.encode(spans, normalize_embeddings=True)
        E_goal = emb.encode(goals, normalize_embeddings=True)
        S_full, S_ans = E_full @ E_full.T, E_ans @ E_ans.T
        n = len(rs)
        pairs = [(spans[j], spans[i]) for i in range(n) for j in range(n) if i != j]
        P = entail(pairs)
        k = 0
        for i, r in enumerate(rs):
            others = [j for j in range(n) if j != i]
            r["sim_full"] = float(S_full[i, others].mean()) if others else None
            r["sim_answer"] = float(S_ans[i, others].mean()) if others else None
            if others:
                into = P[k:k + n - 1]  # pairs (other j, this i): does another node's answer entail this one
                r["nli_agree"] = float(into.mean())
                k += n - 1
            else:
                r["nli_agree"] = None
            r["goal_rel"] = float(E_goal[i] @ E_ans[i])
            chunks = [c[:1500] for c in docs.for_node(t["nodes"][r["node"]]).split("\n") if len(c) > 80][:20]
            sup = entail([(c, spans[i]) for c in chunks])
            r["nli_support"] = float(sup.max()) if len(sup) else None

    print(f"{len(rows)} nodes, {len(by_item)} items\n")
    print("AUROC vs. 'node has the gold answer' (0.5 = chance)")
    kinds = ["all"] + sorted({r["kind"] for r in rows})
    print(f"  {'nodes':8}" + "".join(f"{s:>13}" for s in SIGNALS))
    for kd in kinds:
        xs = rows if kd == "all" else [r for r in rows if r["kind"] == kd]
        line = f"  {kd:8}"
        for s in SIGNALS:
            have = [x for x in xs if x.get(s) is not None]
            a = auroc([x[s] for x in have if x["correct"]], [x[s] for x in have if not x["correct"]])
            line += f"{(f'{a:.2f} ({len(have)})' if a is not None else '—'):>13}"
        print(line)

    # selection: per item, take the answer of the node with the highest score
    ans = {json.loads(l)["id"]: json.loads(l) for l in open(run / "answers.jsonl")}
    single = [i for i in by_item if ans[i]["set"] != "full_system"]
    print(f"\nPick each item's answer from the node with the highest score ({len(single)} single-answer items):")
    base = lambda sysname: np.mean([bool(ans[i][sysname]["correct"]) for i in single])
    print(f"  {'SciWeaver now':26} {base('sciweaver'):.3f}\n  {'react':26} {base('react'):.3f}")
    print(f"  {'oracle (any node correct)':26} {np.mean([any(r['correct'] for r in by_item[i]) for i in single]):.3f}")
    for s in SIGNALS:
        acc = [max((r for r in by_item[i] if r.get(s) is not None and r["op"] != "aggregate"),
                   key=lambda r: r[s], default={"correct": False})["correct"] for i in single]
        print(f"  {'max ' + s:26} {np.mean(acc):.3f}")

    # learned combination, leave-one-item-out
    from sklearn.linear_model import LogisticRegression
    feats = ["agreement", "sim_answer", "nli_agree", "nli_support", "goal_rel", "judge"]
    def X(r):
        v = [r.get(f) for f in feats]
        return [0.0 if x is None else float(x) for x in v] + [float(x is None) for x in v] + [float(r["op"] == "root")]
    accs, probs, labels = [], [], []
    for i in by_item:
        train = [r for j in by_item if j != i for r in by_item[j]]
        clf = LogisticRegression(max_iter=1000, C=1.0).fit([X(r) for r in train], [r["correct"] for r in train])
        test = by_item[i]
        p = clf.predict_proba([X(r) for r in test])[:, 1]
        probs += list(p)
        labels += [r["correct"] for r in test]
        if i in single:
            cand = [(pp, r) for pp, r in zip(p, test) if r["op"] != "aggregate"]
            accs.append(max(cand, key=lambda c: c[0])[1]["correct"] if cand else False)
    a = auroc([p for p, l in zip(probs, labels) if l], [p for p, l in zip(probs, labels) if not l])
    print(f"\nLearned verifier (logistic, leave-one-item-out): AUROC {a:.2f}; pick-best accuracy {np.mean(accs):.3f}")

    (run / "semantic_signals.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"written: {run / 'semantic_signals.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=LOCAL / "runs/pilot_bio_shared_small")
    ap.add_argument("--domain", default="bio")
    a = ap.parse_args()
    main(a.run.expanduser(), a.domain)
