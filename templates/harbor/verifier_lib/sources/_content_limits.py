"""Shared helper for making a source extractor's char-cap truncation visible in its output.

Why this exists: every extractor in this package (xlsx, docx, pptx, pdf, csv, md, text) caps its
returned text at ``context.max_content_chars`` (default 90000, see ``source_types.SourceContext``)
and, before this module existed, did so silently -- ``text[:limit]`` with no signal anywhere in
the returned string or the output model that anything was cut. A judge grading that capped text
has no way to tell "the candidate's real answer ends here" from "the extractor gave up here",
and grades a partial document as if it were the whole submission. An external QC review across
several benchmarks flagged this directly: a 72.6%-discarded reference answer on one task, and one
other task graded with anywhere from 57% to 100% of a given model's own submitted work visible,
depending on how verbose that particular model happened to be -- a bias that falls almost
entirely on the more verbose (and, in that review, lower-scoring) end, not a neutral cap.

This module does not raise the cap and does not fail the run -- both are legitimate alternative
fixes (see each source module's own docstring for why this one was chosen here instead), but
picking a new cap value or deciding a whole eval run should hard-fail is a product/cost decision
this module is not positioned to make unilaterally. What it does instead: whenever a cap actually
truncates something, it appends a short, unmistakable notice INTO the text itself, so whichever
judge (human or LLM) reads the extracted text sees the cut directly in the content it is grading,
rather than a confident-looking document that quietly stops mid-way. This is a deliberate
trade-off, not a full fix -- see each call site for why it was judged sufficient there.

Known limit: on a boundary case where the source's real content ends EXACTLY at `limit`
characters, `truncated` can still read True (the incremental extractors below stop as soon as
they reach or pass the cap, without checking whether anything real was left to add), producing a
harmless false-positive notice on a document that was not actually cut. This is intentional --
the alternative (peeking ahead before appending, on every extractor) is not worth the complexity
for a boundary case that costs nothing worse than one unnecessary but truthful-enough disclosure.
"""

from __future__ import annotations

TRUNCATION_NOTICE_TEMPLATE = (
    "\n\n[TRUNCATED: extraction stopped at the {limit}-character cap for this file -- "
    "content beyond this point was not shown to the grader]"
)


def cap_with_notice(text: str, limit: int) -> str:
    """For extractors that build the full text first and slice it once (csv/md/text): returns
    `text` unchanged if it already fits within `limit`, otherwise a truncated copy with a notice
    appended so the cut is visible in the graded content itself. The notice's own length is
    reserved out of `limit` so the notice itself is never the thing that gets cut off."""
    if len(text) <= limit:
        return text
    notice = TRUNCATION_NOTICE_TEMPLATE.format(limit=limit)
    return text[: max(0, limit - len(notice))] + notice


def finalize_accumulated(parts: list[str], limit: int, truncated: bool) -> str:
    """For extractors that accumulate `parts` incrementally and stop early once a running total
    reaches `limit` (xlsx/docx/pptx/pdf): joins `parts`, applies the same defensive `[:limit]`
    slice these extractors already used, and appends the same truncation notice as
    `cap_with_notice` when the caller reports it stopped early (`truncated=True`) -- see this
    module's docstring for the accepted boundary-case false positive this can produce."""
    text = "\n".join(parts)[:limit]
    if not truncated:
        return text
    notice = TRUNCATION_NOTICE_TEMPLATE.format(limit=limit)
    return text[: max(0, limit - len(notice))] + notice
