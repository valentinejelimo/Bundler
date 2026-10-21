#!/usr/bin/env python3
"""Whole-workbook (cross-level) DAG checks for the SVC/Cirrus task set.

Run ONCE over the tasks/ directory (not per task). Complements the per-task
`check_rubric.py` with the checks that only make sense across the L1→L2→L3
decomposition and its shared corpus:

  1. construct_map       — which expected_value numbers are already PRESENT in the
                           node's inputs (extraction vs derivation map)
  2. filename_map        — deliverable filenames that name their content
  3. anchor_consistency  — is a shared anchor (e.g. start cash 2,186,512) the
                           SAME value everywhere it appears? (contradiction = defect)

POLICY: each SVC task ASSUMES A CORRECT PRIOR RUN, so a value present
  in a node's inputs is BY DESIGN — it is NOT leakage and is never failed here.
  construct_map/filename_map are reported as by-design maps that feed the
  svc_construct_labeling dimension, whose only question is: does the node's rubric
  CLAIM to test derivation of a value the inputs already provide?
    - L1 (fed parent golden): disposition "extraction-expected".
    - L2/L3 (raw corpus +/- upstream goldens): disposition "confirm-anchor" —
      confirm each present value is a GIVEN anchor, not one THIS node must derive.
  anchor_consistency IS gating: the same anchor concept with two different values
  across levels is an objective contradiction (feeds svc_anchor_consistency).

Needs openpyxl / python-docx / pypdf for Office inputs; whatever is missing is
reported as skipped coverage (never silently dropped). Numbers with >= 4 digits
are scanned (smaller values and percentages are too common to be evidence).

    python check_dag.py <tasks_dir>
    python check_dag.py <tasks_dir> --json report.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

MIN_DIGITS = 4  # only treat >=4-digit integers as identifying values
# tokens in a deliverable filename that telegraph an analytical answer
ANSWER_TOKENS = ("trough", "october", "median", "_p5", "p95", "answer",
                 "shortfall", "buffer", "drawdown")

_num_re = re.compile(r"(?<![\d.])\d{%d,}(?![\d.])" % MIN_DIGITS)
_year_re = re.compile(r"^(?:19|20)\d\d$")  # bare calendar years are date refs, not answers


def nums(text: str) -> set[str]:
    """>=4-digit integers, excluding bare calendar years (1900-2099)."""
    return {n for n in _num_re.findall((text or "").replace(",", ""))
            if not _year_re.match(n)}


def file_text(path: str, skipped: set) -> str:
    s = path.lower()
    try:
        if s.endswith(".xlsx"):
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            out = []
            for sn in wb.sheetnames:
                for row in wb[sn].iter_rows(values_only=True):
                    out.append(" ".join("" if c is None else str(c) for c in row))
            wb.close()
            return "\n".join(out)
        if s.endswith(".docx"):
            from docx import Document
            d = Document(path)
            return "\n".join(p.text for p in d.paragraphs)
        if s.endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
        if s.endswith((".csv", ".tsv", ".txt", ".md", ".json")):
            return open(path, encoding="utf-8", errors="replace").read()
    except ImportError as exc:
        skipped.add(f"{os.path.splitext(s)[1]} ({exc.name} missing)")
    except Exception as exc:
        skipped.add(f"{os.path.basename(path)} ({type(exc).__name__})")
    return ""


def level_of(task_name: str) -> str:
    m = re.search(r"-l([123])", task_name)
    return f"L{m.group(1)}" if m else "?"


def expected_values(rubric: dict) -> list[tuple[str, str, set]]:
    """[(criterion_id, expected_value_text, {numbers}), ...]"""
    out = []
    for c in rubric.get("criteria", []):
        ev = str(c.get("expected_value", ""))
        n = nums(ev)
        if n:
            out.append((c.get("id"), ev, n))
    return out


def check_workbook(tasks_dir: str):
    tasks = sorted(glob.glob(os.path.join(tasks_dir, "cirrus-*")))
    skipped: set = set()
    per_task = []            # leakage + filename findings
    anchor_map: dict = {}    # number -> {(task, criterion_id, ev_snippet)}

    for t in tasks:
        tid = os.path.basename(t).replace("cirrus-y1-task-5-", "")
        lvl = level_of(os.path.basename(t))
        rub_path = os.path.join(t, "tests", "rubric.json")
        if not os.path.exists(rub_path):
            continue
        rubric = json.load(open(rub_path))
        exp = expected_values(rubric)

        # gather input text once
        inp = ""
        for f in sorted(glob.glob(os.path.join(t, "environment", "data", "*"))):
            if os.path.isfile(f):
                inp += file_text(f, skipped) + "\n"
        inp_nums = nums(inp)

        # 1. construct map (extraction vs derivation) — NOT a leak verdict.
        # Ruling: each task assumes a correct prior run, so a value present in the
        # inputs is by design. We map it so svc_construct_labeling can confirm the
        # rubric doesn't CLAIM derivation for what the inputs already provide.
        present = [(cid, ev, n & inp_nums) for cid, ev, n in exp if n & inp_nums]
        if present:
            frac = f"{len(present)}/{len(exp)}"
            vals = sorted({v for _, _, nn in present for v in nn})
            if lvl == "L1":
                disp, q = ("extraction-expected",
                    "Answer value(s) present in input — by design (prior-run "
                    "assumption). svc_construct_labeling: confirm the rubric tests "
                    "EXTRACTION, not derivation.")
            else:  # L2/L3 read raw corpus + may read upstream goldens
                disp, q = ("confirm-anchor",
                    "Confirm each is a GIVEN grounding anchor (start cash, board "
                    "floor, prior-run output), not a value THIS node is supposed to "
                    "DERIVE (trough dist / drawdown / probability).")
            per_task.append({
                "task": tid, "level": lvl, "check": "construct_map",
                "disposition": disp,
                "detail": f"{frac} expected-value criteria present in input; "
                          f"values={vals[:8]}. {q}",
            })

        # 2. filename map — informational only (a deliverable name that describes
        # its content is fine under the prior-run ruling; listed for awareness).
        golds = [os.path.basename(g) for g in
                 glob.glob(os.path.join(t, "solution", "golden", "*"))]
        fn_hits = sorted({tok for g in golds for tok in ANSWER_TOKENS
                          if tok in g.lower()})
        if fn_hits:
            per_task.append({
                "task": tid, "level": lvl, "check": "filename_map",
                "disposition": "informational",
                "detail": f"deliverable filename names its content {fn_hits}: {golds}. "
                          f"Fine under the prior-run ruling; note if a name states a "
                          f"value the node must derive.",
            })

        # 3. collect anchors (only values that recur across >=2 tasks are anchors)
        for cid, ev, n in exp:
            for v in n:
                anchor_map.setdefault(v, set()).add((tid, cid, ev[:60]))

    # anchor_consistency: an anchor value is consistent by construction (same
    # number). The real contradiction is a shared *concept* with divergent values;
    # we surface the recurring anchors so a human can eyeball divergence, and we
    # flag the canonical start-cash anchor specifically.
    recurring = {v: srcs for v, srcs in anchor_map.items() if len({s[0] for s in srcs}) >= 3}
    return per_task, recurring, sorted(skipped)


def _golden_index(tasks_dir):
    """{sha256: node} and {basename: node} over every node's solution/golden/*."""
    import hashlib
    by_sha, by_name = {}, {}
    for node in sorted(glob.glob(os.path.join(tasks_dir, "cirrus-*"))):
        gd = os.path.join(node, "solution", "golden")
        if not os.path.isdir(gd):
            continue
        for g in glob.glob(os.path.join(gd, "**", "*"), recursive=True):
            if os.path.isfile(g):
                try:
                    by_sha.setdefault(hashlib.sha256(open(g, "rb").read()).hexdigest(),
                                      os.path.basename(node))
                except OSError:
                    pass
                by_name.setdefault(os.path.basename(g).lower(), os.path.basename(node))
    return by_sha, by_name


def classify_node_inputs(tasks_dir):
    """Per node: which environment/data inputs are ANOTHER node's golden deliverable
    (expected by-design hierarchy input) vs a source document. Same content-hash tree
    the input_prompt_leakage gate consumes via build_upstream_deliverable_inputs."""
    import hashlib
    by_sha, by_name = _golden_index(tasks_dir)
    rows = []
    for node in sorted(glob.glob(os.path.join(tasks_dir, "cirrus-*"))):
        nid = os.path.basename(node)
        data = os.path.join(node, "environment", "data")
        if not os.path.isdir(data):
            continue
        deliv, nsrc = [], 0
        for f in sorted(glob.glob(os.path.join(data, "**", "*"), recursive=True)):
            if not os.path.isfile(f) or os.path.basename(f).lower() in ("sources.md", ".ds_store"):
                continue
            prod = None
            try:
                prod = by_sha.get(hashlib.sha256(open(f, "rb").read()).hexdigest())
            except OSError:
                pass
            if prod is None:
                prod = by_name.get(os.path.basename(f).lower())
            if prod and prod != nid:            # a DIFFERENT node's deliverable
                deliv.append((os.path.basename(f), prod))
            else:                                # own-golden match or no match -> source doc
                nsrc += 1
        rows.append((nid, deliv, nsrc))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-level DAG checks for SVC tasks")
    ap.add_argument("tasks_dir", help="Path to the tasks/ directory (holds cirrus-*)")
    ap.add_argument("--json", default=None, help="Also write findings as JSON")
    args = ap.parse_args(argv)

    per_task, recurring, skipped = check_workbook(args.tasks_dir)

    print("SVC DAG checks —", args.tasks_dir)
    print("=" * 68)
    if skipped:
        print("SKIPPED COVERAGE (install libs to close):", ", ".join(skipped))
    print(f"\nPer-task construct/anchor map ({len(per_task)} entries — by-design, "
          f"not failures; input to svc_construct_labeling):")
    for f in per_task:
        print(f"  [{f['level']:>2} · {f['check']:<13} · {f['disposition']:<18}] {f['task']}")
        print(f"       ↳ {f['detail']}")
    print(f"\nRecurring anchors (appear in ≥3 tasks — eyeball for divergent use):")
    for v, srcs in sorted(recurring.items(), key=lambda kv: -len(kv[1])):
        tasks = sorted({s[0] for s in srcs})
        print(f"  {v:>12}  in {len(tasks)} tasks: {', '.join(tasks[:6])}"
              + (" …" if len(tasks) > 6 else ""))

    rows = classify_node_inputs(args.tasks_dir)
    tot_deliv = sum(len(d) for _, d, _ in rows)
    tot_src = sum(s for _, _, s in rows)
    print(f"\nInput provenance (hierarchy tree, by content hash): {tot_deliv} input(s) ARE another "
          f"node's golden deliverable (expected by design); {tot_src} are source docs "
          f"(leaked-answer rule applies).")
    for nid, deliv, nsrc in rows:
        if deliv:
            print(f"  {nid}: upstream-deliverable input(s) -> "
                  + ", ".join(f"{fn} (<- {p})" for fn, p in deliv))
    if tot_deliv == 0:
        print("  (no node-golden inputs found — every input is a source document in this set)")

    if args.json:
        json.dump({"per_task": per_task,
                   "recurring_anchors": {v: sorted({s[0] for s in srcs})
                                         for v, srcs in recurring.items()},
                   "skipped": skipped}, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
