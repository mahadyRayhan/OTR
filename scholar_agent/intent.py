"""
intent.py — does a request ask the scholar agent for a paper (its identity or DOI) rather than for experts?

Shared by the decomposer (routing, no "Find N experts" prefix) and the scholar agent (paper mode).
Requests about a paper's content ("what does the X paper report") are not lookups; they go to RAG.
"""

from __future__ import annotations

import re

PAPER_LOOKUP = re.compile(
    r"\bDOIs?\b"
    r"|\b(?:which|identify|find|locate|search\s+for|look\s+up|name|cite)\b[^.?!\n]{0,60}?"
    r"\b(?:papers?|publications?|articles?|preprints?)\b"
    r"|\b(?:papers?|publications?|articles?|preprints?)\s+(?:that|which)\s+(?:first\s+)?"
    r"(?:introduc|describ|report|present|propos)\w*"
    # "retrieve the paper ..." asks for the paper itself; "retrieve X from the paper" (content) does not match.
    r"|\b(?:retrieve|get|fetch|obtain)\s+(?:the|a)\s+(?:[\w-]+\s+){0,2}(?:papers?|publications?|articles?|preprints?)\b",
    re.IGNORECASE,
)
EXPERT_REQUEST = re.compile(r"\b(?:experts?|researchers?|collaborat\w*|scientists?|who\s+(?:works?|is|are))\b",
                            re.IGNORECASE)


def is_paper_lookup(text: str) -> bool:
    return bool(PAPER_LOOKUP.search(text or "")) and not EXPERT_REQUEST.search(text or "")
