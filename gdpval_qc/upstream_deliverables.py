"""Classify a task's input files as registered UPSTREAM NODE DELIVERABLES vs
source documents, for hierarchical (L1/L2/L3) benchmarks.

The benchmark tree IS the task subdirectories: each node's ``solution/golden/*``
is that node's deliverable. An input file under ``environment/data/`` that
byte-matches (sha256) — or, as a weaker fallback, exact-name-matches — some
*other* node's golden is that node's deliverable legitimately appearing
downstream: expected by design, NOT a leak. Everything else is a source
document, subject to the normal leaked-answer / contamination rule.

Consumed two ways:
  - check_dag.py     : workbook-level report (deliverable-inputs vs source-docs)
  - loader callback  : `build_upstream_deliverable_inputs` feeds the
                       input_prompt_leakage dimension so the gate can tell the
                       two apart deterministically instead of by LLM guess.

Note: an input that matches a node's OWN golden is NOT counted as an upstream
deliverable — that would be a direct answer leak and must still be flagged.
Renamed / re-exported copies of a deliverable (different bytes AND different
name) are not auto-detected; feed upstream deliverables verbatim so the sha
matches.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Files that are never task inputs (pointers / OS noise).
_SKIP = {"sources.md", ".ds_store"}

_index_cache: dict[str, tuple[dict, dict]] = {}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def golden_index(corpus_root: Path) -> tuple[dict, dict]:
    """{sha: node_id}, {basename_lower: node_id} for every node's goldens."""
    key = str(corpus_root)
    if key in _index_cache:
        return _index_cache[key]
    by_sha: dict[str, str] = {}
    by_name: dict[str, str] = {}
    try:
        nodes = sorted(p for p in corpus_root.iterdir() if p.is_dir())
    except OSError:
        nodes = []
    for node in nodes:
        gd = node / "solution" / "golden"
        if not gd.is_dir():
            continue
        for g in gd.rglob("*"):
            if g.is_file():
                try:
                    by_sha.setdefault(_sha(g), node.name)
                except OSError:
                    pass
                by_name.setdefault(g.name.lower(), node.name)
    _index_cache[key] = (by_sha, by_name)
    return by_sha, by_name


def classify_inputs(task_dir: Path):
    """Return (deliverable_inputs, source_inputs).

    deliverable_inputs: list of (filename, producing_node) for inputs that are a
    *different* node's golden. source_inputs: list of filenames for the rest.
    """
    task_dir = Path(task_dir)
    by_sha, by_name = golden_index(task_dir.parent)
    data = task_dir / "environment" / "data"
    deliverable, source = [], []
    if data.is_dir():
        for f in sorted(data.rglob("*")):
            if not f.is_file() or f.name.lower() in _SKIP:
                continue
            prod = None
            try:
                prod = by_sha.get(_sha(f))
            except OSError:
                pass
            if prod is None:
                prod = by_name.get(f.name.lower())
            if prod and prod != task_dir.name:      # a DIFFERENT node's deliverable
                deliverable.append((f.name, prod))
            else:                                     # own-golden match or no match -> source doc
                source.append(f.name)
    return deliverable, source


def build_upstream_deliverable_inputs(sample) -> str:
    """Loader callback → text note of the sample's registered upstream-deliverable
    inputs (empty string when there are none)."""
    try:
        deliverable, _ = classify_inputs(sample.task_dir)
    except Exception:
        return ""
    if not deliverable:
        return ""
    lines = "\n".join(f"- {fn}  (deliverable of node {node})" for fn, node in deliverable)
    return (
        "Registered UPSTREAM-DELIVERABLE inputs — provided by design in this "
        "hierarchical benchmark; these are NOT leaks and must not be flagged as "
        "'completed document in inputs':\n" + lines
    )
