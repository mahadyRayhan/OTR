"""
lookups.py — database lookups agents and verifiers can rely on: NCBI E-utilities and remote BLAST.

They run on the host, not in the sandbox (which has no network), against fixed endpoints only. Each
call is rate-limited to NCBI's rules (3 requests/s; 10/s with NCBI_API_KEY), cached while the run cache
is on (llm_cache.memo, so a replay sees the same records), and recorded in the run trace.

  esearch(db, term, retmax)         -> list of ids          e.g. esearch("gene", "BRCA1[sym] AND human[orgn]")
  esummary(db, ids)                 -> {id: summary dict}
  efetch(db, ids, rettype, retmode) -> text                 e.g. efetch("nuccore", ["NM_007294"], "fasta")
  blast(sequence, program, database, hitlist_size) -> [{"accession", "title", "evalue", "identity", ...}]

Errors raise LookupFailed with the reason, so a caller can tell "not found" (an empty result) from
"could not ask" (an exception).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request

from orchestrator import llm_cache, trace

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
TOOL = "sciweaver"

_lock = threading.Lock()
_last = 0.0


class LookupFailed(RuntimeError):
    pass


def _wait() -> None:
    """NCBI allows 3 requests/s without an API key and 10/s with one."""
    global _last
    gap = 0.11 if os.environ.get("NCBI_API_KEY") else 0.34
    with _lock:
        now = time.time()
        if now - _last < gap:
            time.sleep(gap - (now - _last))
        _last = time.time()


def _get(url: str, params: dict, timeout: int = 30) -> str:
    params = {**params, "tool": TOOL}
    if os.environ.get("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    _wait()
    try:
        with urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(params)}", timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise LookupFailed(f"{url}: {e}") from e


def _traced(tool: str, args: dict, fn):
    t0 = time.time()
    try:
        out = llm_cache.memo(tool, args, fn)
    except LookupFailed as e:
        trace.record_tool_call(tool, args, {"error": str(e)[:200], "seconds": round(time.time() - t0, 2)})
        raise
    size = len(out) if hasattr(out, "__len__") else None
    trace.record_tool_call(tool, args, {"results": size, "seconds": round(time.time() - t0, 2)})
    return out


def esearch(db: str, term: str, retmax: int = 20) -> list[str]:
    def run():
        data = json.loads(_get(EUTILS + "esearch.fcgi", {"db": db, "term": term, "retmax": retmax, "retmode": "json"}))
        return data.get("esearchresult", {}).get("idlist", [])
    return _traced("ncbi.esearch", {"db": db, "term": term, "retmax": retmax}, run)


def esummary(db: str, ids: list[str]) -> dict:
    if not ids:
        return {}

    def run():
        data = json.loads(_get(EUTILS + "esummary.fcgi", {"db": db, "id": ",".join(ids), "retmode": "json"}))
        res = data.get("result", {})
        return {i: res[i] for i in res.get("uids", [])}
    return _traced("ncbi.esummary", {"db": db, "ids": list(ids)}, run)


def efetch(db: str, ids: list[str], rettype: str = "fasta", retmode: str = "text") -> str:
    if not ids:
        return ""
    return _traced("ncbi.efetch", {"db": db, "ids": list(ids), "rettype": rettype, "retmode": retmode},
                   lambda: _get(EUTILS + "efetch.fcgi", {"db": db, "id": ",".join(ids), "rettype": rettype,
                                                         "retmode": retmode}, timeout=60))


def blast(sequence: str, program: str = "blastn", database: str = "nt", hitlist_size: int = 5,
          max_wait: int = 300) -> list[dict]:
    """NCBI remote BLAST (URL API: submit, poll, fetch). Slow: usually 20 s to a few minutes."""
    seq = re.sub(r"\s+", "", sequence)

    def run():
        put = _get(BLAST_URL, {"CMD": "Put", "PROGRAM": program, "DATABASE": database, "QUERY": seq,
                               "HITLIST_SIZE": hitlist_size, "FORMAT_TYPE": "JSON2_S"})
        m = re.search(r"RID = (\S+)", put)
        if not m:
            raise LookupFailed("BLAST: no request id returned")
        rid, start = m.group(1), time.time()
        while True:
            time.sleep(10)
            status = _get(BLAST_URL, {"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"})
            if "Status=READY" in status:
                break
            if "Status=FAILED" in status or "Status=UNKNOWN" in status:
                raise LookupFailed(f"BLAST request {rid} failed")
            if time.time() - start > max_wait:
                raise LookupFailed(f"BLAST request {rid} not ready after {max_wait} s")
        data = json.loads(_get(BLAST_URL, {"CMD": "Get", "RID": rid, "FORMAT_TYPE": "JSON2_S"}, timeout=60))
        hits = []
        for h in data["BlastOutput2"][0]["report"]["results"]["search"].get("hits", []):
            d, hsp = h["description"][0], h["hsps"][0]
            hits.append({"accession": d.get("accession"), "title": d.get("title"), "taxid": d.get("taxid"),
                         "evalue": hsp.get("evalue"), "identity": hsp.get("identity"),
                         "align_len": hsp.get("align_len"), "bit_score": hsp.get("bit_score")})
        return hits
    return _traced("ncbi.blast", {"program": program, "database": database, "sequence": seq,
                                  "hitlist_size": hitlist_size}, run)
