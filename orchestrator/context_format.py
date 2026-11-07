"""
context_format.py — recover the task from a sub-task string that carries upstream context.

engine.py hands agents one string. With context routing it is either

  <context blocks>---\n## Your Task\n<task>                       (CUGCR, build_rigid_prompt)
  ⚠️ LOW CONFIDENCE CONTEXT ...\n\n<task>\n\n[UPSTREAM CONTEXT (truncated)]:\n<context>   (CRS gate)

Agents should search their indexes with the task, not the context: the embedding model reads only
the first few hundred tokens, so a context-first string retrieves documents about the context.

engine.py also appends the user's original request (with_request): goals refer to "the provided
sequence" or "options A-D", which only the request contains. It is reference material, not part of
the task, so retrieval drops it too (without_request).
"""

from __future__ import annotations

import re

REQUEST_HEADER = "### Original user request"
# A goal that points at material in the request: "the provided sequence", "the given values", "options A-D".
REFERS_TO_REQUEST = re.compile(
    r"\b(?:provided|given|supplied|following|above|options?|choices|the\s+(?:user'?s?\s+)?(?:query|request|question))\b",
    re.IGNORECASE)


def with_request(subtask: str, request: str) -> str:
    """Append the user's request when the goal refers to material in it (data, options, names).

    Only then: the request also carries the final answer's format, which sub-tasks must not follow.
    """
    task = task_text(subtask)
    if not request or request.strip() in subtask or not REFERS_TO_REQUEST.search(task):
        return subtask
    return (f"{subtask}\n\n{REQUEST_HEADER} (reference only: use it for the data, options or names your task "
            f"refers to. Do only your task; any answer format it asks for applies to the final answer, not to "
            f"you)\n{request.strip()}")


def without_request(subtask: str) -> str:
    return (subtask or "").split(REQUEST_HEADER, 1)[0].strip()


def task_text(subtask: str) -> str:
    """The task part of an engine sub-task, with any '--strategy=' flag removed."""
    text = without_request(subtask)
    if "## Your Task" in text:
        text = text.split("## Your Task", 1)[1]
    elif text.lstrip().startswith("⚠️ LOW CONFIDENCE CONTEXT"):
        text = text.split("\n\n", 1)[1] if "\n\n" in text else text
    text = text.split("[UPSTREAM CONTEXT", 1)[0]
    return re.sub(r"--strategy=\w+", "", text).strip()


def entities(subtask: str) -> str:
    """The 'Key Technical Entities' line of the context, if any (tool names, accessions, ...)."""
    m = re.search(r"### Key Technical Entities\n(.+)", subtask or "")
    return m.group(1).strip() if m else ""
