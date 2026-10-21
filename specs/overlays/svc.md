---
name: gdpval_svc_qc
extends: ../gdpval_task_quality_v18.md
# SVC layout remap vs base: input=environment/data, golden=solution/golden,
# verifier_spec_json=tests/rubric.json, verifier_assertion_summary=evals/ digest.
# The 6 LLM verifier dims run (they read the rubric text + the eval summary);
# the 3 PROGRAMMATIC verifier checks are re-gated to task_has_rubric_verifier so
# they skip on SVC (no verifiers[] array). Structural rubric QC: check_rubric.py.
field_types:
  upstream_deliverable_inputs: text # inputs that are another node's golden (hierarchy tree)
field_bindings:
  input_files:
    source: directory
    path: environment/data          # SVC inputs live here (NOT all of environment/)
  solution_files:
    source: directory
    path: solution/golden           # SVC golden dir (was solution/golden_outputs)
    exclude: ["Reasoning.docx"]
  solution_reasoning:
    source: file
    path: solution/golden/Reasoning.docx
    optional: true
  verifier_spec_json:
    source: file
    path: tests/rubric.json         # SVC rubric text (criteria[]); read by the LLM verifier dims
    optional: true
  upstream_deliverable_inputs:
    source: callback
    callback: build_upstream_deliverable_inputs  # inputs that byte-match another node's golden
    optional: true
---

### overall_gdpval_fit

```yaml
dimension_name: "Overall GDPVal Fit"
check_kind: llm
inputs: [instruction, category, solution_files]
applies_when:
  builtin: task_is_substantive      # SVC: skip atomic L1 leaves (false-positive here)
gate:
  role: blocking
  reject_at_severity: 2          # sev 2 = "benchmark-shaped"; not fixable by editing
  on_disagreement: review
```

**What this evaluates.** Holistically, does this task represent realistic,
economically valuable knowledge-work that a professional would be paid to perform?

**How to judge.** This is a single gestalt judgment, not a sum of subcriteria.
Take a top-down impression of `instruction` + `category` + the golden solution
together. The question is only: if you encountered this task in a real
professional workflow, would you immediately recognise it as legitimate paid work
in that domain? Other dimensions handle authenticity, complexity, and formatting
in detail; do **not** re-grade those concerns here.

**Severity rubric.**
- **0** — Unmistakably real delegated professional work. The domain is obvious,
  the golden solution demonstrates genuine economic value, and the scope feels
  appropriate for a billable engagement.
- **1** — Reads as real professional work with a mild constructed feel — slightly
  oversized or undersized scope, slightly idealised stakeholders, or stakes that
  feel a touch hypothetical.
- **2** — Materially test-like. Individual sub-components are plausible, but the
  whole reads as a benchmark or training exercise (round-numbered financials,
  generic "Acme Corp" stakeholders, scope chosen to exercise specific
  capabilities).
- **3** — Clearly constructed and not credible as real delegated work — template
  structure, "illustrative" framings, or textbook-shaped scenarios.

**Common hard-fail patterns.**
- Task framing is paper-thin — a layer of professional role over what is
  fundamentally a benchmark question (the role and stakes are decorative,
  not driving the content).
- Instruction enumerates the deliverable's required sections in language that
  reads as a verifier rubric copied verbatim into the prompt (e.g. each
  section header maps 1-to-1 to a rubric item, often using the rubric's
  internal phrasing). Normal "the memo should cover X, Y, and Z" enumeration
  is **not** a hard-fail.
- Arbitrary, benchmark-shaped constraints with no plausible workplace
  motivation: "exactly 3 bullet points", "exactly 200 words". Natural
  professional length cues like "a one-page memo" or "a ~500-word summary"
  are **fine**.

**Anti-overcall.** Real professional work is often structured, repetitive,
and dryly written. Eval-context conventions (persona prompts, output file
paths, named deliverables, required-format specifications, length cues)
are not synthetic markers — see the spec intro for the full list of
eval-context conventions. Flag only when the *overall feel* of the sample
reads as exercise-shaped after holistic consideration.

**SVC note.** This dim is skipped on L1 leaves (atomic decomposition nodes, via
`task_is_substantive`) — an atomic sub-task is not standalone billable work by design.
When it runs (L2/L3), do NOT down-grade fit for cross-reference / provenance markers in the
golden (e.g. `transcribe the verified L3 engine`, `ties to the L3-05B golden`, `Golden databook`);
those are intended decomposition data-flow references, not "testing artifacts" that make the work
unreal. Judge the underlying work's realism, not the provenance headers.

---

### input_prompt_leakage

```yaml
dimension_name: "Input Folder Prompt Leakage"
check_kind: llm
inputs: [instruction, input_files, upstream_deliverable_inputs]
gate:
  role: blocking
  reject_at_severity: 2
  on_disagreement: review
```

**What this evaluates.** Whether any file in the input folder (`environment/`) is a copy
or near-duplicate of the task instruction, or is a solution/reasoning artifact (e.g. a
`Reasoning.docx`, a completed deliverable, or a grading rubric) that should not be
visible to the agent. This is a contamination check, not a consistency check — it is
asking whether the folder accidentally gives the agent the answer or the prompt itself.

**Severity rubric.**

- **Severity 0** — All input files are genuine source documents. No file is semantically
  equivalent to the instruction or contains information that would constitute a direct
  answer leak. The folder is clean.
- **Severity 1** — One file has minor overlap with the instruction (e.g. shares a
  sentence of boilerplate context) but is clearly a legitimate source document in its
  own right. The overlap does not materially help the agent bypass the task.
- **Severity 2** — One or more input files is a near-duplicate of the instruction, a
  copy of the task prompt, a completed solution deliverable, or a reasoning/justification
  document that reveals how the task should be solved. The contamination materially
  undermines the task's integrity as an evaluation item.
- **Severity 3** — The input folder contains the full prompt verbatim AND one or more
  solution artifacts, making the task entirely transparent to the agent.

**Common hard-fail patterns (sev 2).**

- A file named `Prompt.docx`, `prompt.md`, `instruction.docx`, `task.md`, or any
  filename that explicitly identifies the file as the task instruction.
- A file whose opening paragraph is identical or near-identical to the first paragraph
  of the instruction (copy-paste of the prompt into a Word doc).
- A file named `Reasoning.docx` or `Reasoning.pdf` in the input folder — this is the
  canonical name for the expert's solution reasoning and must only appear under
  `solution/golden_outputs/`.
- A completed deliverable (e.g. a finished memo, a completed financial model with
  specific conclusions) placed in the input folder alongside the raw source data.
- A rubric or scoring guide that lists the verifier criteria, placed in the input folder.

**Anti-overcall guidance.**

- Source documents that reference the same company, transaction, or scenario as the
  instruction are expected and must NOT be flagged — the instruction describes the
  task context, and the source docs naturally share that context.
- Partial thematic overlap (e.g. the instruction mentions "leverage ratio" and a
  source document is a credit agreement that discusses leverage) is not leakage.
- Files named `background.docx`, `context.pdf`, `briefing.md` are only suspicious if
  their *content* mirrors the instruction — the filename alone is not sufficient to flag.
- If a file looks like a legitimate professional document with genuine domain content,
  do not flag it simply because it was named generically.
- Evidence quote is required for any severity ≥ 1 finding. Quote the specific passage
  from the input file that is equivalent to instruction content.
- **SVC decomposition-DAG exception.** Many SVC tasks are decomposition nodes designed to
  *consume an upstream deliverable*: the instruction explicitly provides a prior-run databook
  (e.g. `L2-01_Baseline_Cash_Databook.xlsx`) as the input to read. When the instruction points the
  agent at such an upstream analysis/databook as its designated input, that is the intended data
  flow — NOT contamination — even when the databook contains the exact figure the task asks to
  report (each task assumes a correct prior run). Do not flag it. Continue to flag genuine
  accidents: a verbatim copy of THIS task's instruction, a `Reasoning.docx`, or a completed
  deliverable/rubric that was not provided as a designated input.
- **Deterministic deliverable list.** The `upstream_deliverable_inputs` field lists (by content
  hash, from the benchmark tree) exactly which input files ARE another node's golden deliverable —
  those are the by-design hierarchy inputs; **never flag a file named there.** A file NOT on that
  list is a source document: apply the leaked-answer rule to it normally (a source doc that hands
  over this task's answer, or carries grading / pipeline / AI-provenance metadata, is still a real
  leak — the deliverable exemption does not cover source docs).

---

### professional_complexity

```yaml
dimension_name: "Professional Complexity"
check_kind: llm
inputs: [instruction, input_files, solution_files]
applies_when:
  builtin: task_is_substantive      # SVC: L1 leaves are atomic-by-design
gate:
  role: blocking
  reject_at_severity: 2          # sev 2 = "predominantly mechanical"; can't add judgment by editing
  on_disagreement: review
```

**What this evaluates.** Does the task require genuine professional judgment —
synthesis across documents, weighing tradeoffs, applying domain expertise — or
could a non-expert produce the same answer by mechanically following a template?

**How to judge.** Look for tasks where the solution requires actual *decisions*:
deciding what to include and exclude, weighing competing considerations, inferring
intent from ambiguous inputs, or producing analysis that goes beyond
transcription. The golden solution is the strongest signal — does it demonstrate
analytical effort, or is it largely formatting + paraphrase + arithmetic?

**Severity rubric.**
- **0** — Requires meaningful professional judgment. Synthesis, tradeoff
  weighing, or genuine domain expertise visibly drive the solution.
- **1** — Mostly substantive but with some rote / template-able parts (e.g. a
  standard cover memo wrapping the real analysis).
- **2** — Predominantly mechanical with thin layers of judgment — most of the
  work is filling slots in a template, copying numbers, or applying a fixed
  rubric.
- **3** — A first-week intern with a template could produce a comparable
  result.

**Common hard-fail patterns.**
- Pure transcription — task amounts to rewriting one document's content in a
  different format with no analytical work added.
- Pure arithmetic with no judgment ("sum these numbers", "calculate this
  ratio", "convert these units") — the answer is a calculation, not a decision.
- Task reduces to mechanically applying a clearly stated formula or procedure
  the instruction itself spells out.
- The solution could be generated by a 100-line script with no model
  reasoning — i.e. the work is fully procedural.

**Anti-overcall.** Volume of work does not equal complexity, but neither does
brevity equal triviality. Short solutions can require deep judgment (a one-page
recommendation memo, a single chart with a key insight). Judge the *type* of
work the task requires, not its length. A pre-specified deliverable structure
(named files, required sections, length cues) is eval-context convention and
does **not** indicate the task is mechanical — what matters is whether the
*content* requires real judgment.

**SVC calibration (do NOT over-flag).** (1) **Monthly phasing exists** — do not claim the inputs
"lack phasing/seasonality" for a monthly forecast: FY2021 monthly data lives in `Year_1_KPI_Report`
(monthly trends), `Year_1_Annual_PnL` (monthly spine), and `Year_1_Cash_Flow_Statement` (12-month
model); verify those before flagging infeasibility. (2) **Disclosed-assumption underspecification is
by-design** — a probabilistic/estimative prompt that says to "make a sensible assumption and say so"
INVITES the analyst to choose and disclose parameters (volatilities, correlations, shock odds).
Missing up-front parameters is not infeasibility when the task explicitly delegates the choice.

---

### instruction_authenticity

```yaml
dimension_name: "Instruction Authenticity"
check_kind: llm
inputs: [instruction]
gate:
  role: blocking
  reject_at_severity: 2          # sev 2 = "LLM-shaped instruction"; not fixable
  on_disagreement: review
```

**What this evaluates.** Is the instruction text plausibly written by a real
human professional in the relevant domain, or does it read as LLM-generated
or template-filled?

**How to judge.** Look for human signals (domain-specific shorthand, casual
asides, sender-recipient context, references to past communications, dated
items, specific people/places that ground the request) and LLM signals
(hedging language; generic phrasings without grounding; over-clean
numerical examples; overly explicit politeness; scaffolded structure with
no real-world specifics).

A crucial framing for this corpus: the dataset is an *evaluation*, so several
patterns that would be unusual in an in-the-wild email are perfectly
acceptable here. Specifically, none of the following are signals of synthetic
origin and must not be penalised:

- A persona / role assignment ("You are a senior credit analyst", "Acting as
  the portfolio manager, …")
- Output file or path specifications ("save your memo as
  `/workspace/output/memo.docx`", "produce two files named …")
- A required deliverable list, even when enumerated
- Natural length cues ("a one-page memo", "a ~500-word summary")
- Standard business words ("comprehensive", "actionable") used in normal
  proportion
- Anonymised-real organisations or people

**Severity rubric.**
- **0** — Reads unmistakably as written by a human domain practitioner. Specific
  details, natural register, contextual asides.
- **1** — Mostly human but with small stylistic tells suggesting light AI
  assistance.
- **2** — Materially LLM-shaped: hedging without specifics, template-like
  organisation, polished "ask" disconnected from any real-world context.
- **3** — Clearly LLM-generated; reads like prompt output (e.g. "as an AI…",
  systematic paragraphing with each paragraph hitting a different "topic").

**Common hard-fail patterns.**
- Generic placeholder organisations used throughout with no real-world
  anchoring ("ABC Corporation", "Widget Inc", "Generic Industries", "Company
  X"). Anonymised real names are fine; transparently invented stand-ins are
  not.
- Stakeholders named but given no titles, no relational context, no
  organisational structure — just first-name characters appearing without
  backstory or job function.
- Markdown formatting in the instruction body (visible `**bold**`, `-`
  bullets rendered as bullets, `#` headings) when the instruction is
  presented as prose — real professional emails and briefs rarely use raw
  Markdown. **SVC exception:** the harness appends a standard `## Environment
  notes` block (a bulleted source-file list + a "Save your deliverable(s) to
  …" spec) to every SVC instruction; that block is harness-injected boilerplate,
  NOT the expert's prose — do NOT treat its Markdown as a synthetic tell. Judge
  authenticity on the actual brief (the From/To/Subject request) above it.
- "As an AI assistant…" / "I am a language model…" / similar meta-framings.

**Anti-overcall.** Humans *can* write clean, structured instructions —
structure alone is not a tell. Persona prompts, output-file specs, named
deliverables, and required-format lists are eval-context conventions, **not**
synthetic markers (see the spec intro). The "As a {role}, I want to …"
user-story format from agile/software contexts is borderline: prefer
flagging only when the rest of the instruction is also clearly LLM-shaped.
Clusters of business words ("comprehensive", "thorough", "actionable") are
common in normal writing and are weak signals on their own; treat them as
contributing evidence at most, not a hard-fail by themselves. Quote
concrete passages when flagging severity ≥ 1.

---

### input_file_authenticity

```yaml
dimension_name: "Input File Authenticity"
check_kind: llm
inputs: [input_files]
gate:
  role: blocking
  reject_at_severity: 2          # sev 2 = "synthetic feel"; not fixable
  on_disagreement: review
```

**What this evaluates.** Do the input files look like real artifacts produced
by humans in a real professional workflow, or do they look synthesised to
look-the-part?

**How to judge.** Benchmark and evaluation inputs are legitimately synthetic.
They are anonymised for privacy, often use placeholder names, round or projected
figures, and shared templates. None of that is an authenticity defect on its
own, because eval data is required to be synthetic. Judge instead whether the
document is internally coherent and plausible as the artifact it claims to be.
The defects that matter are hard fabrication signals: internal contradictions,
impossible or implausible values, content that could not come from the stated
source, leaked answer data, or task-pipeline markers.

**Severity rubric.**
- **0** — Authentic, or legitimately synthetic and anonymised data that is
  internally coherent and plausible as the claimed artifact.
- **1** — Soft synthetic markers only: generic or placeholder names, round
  numbers, clean or templated formatting, uniform voice across documents, or
  benchmark-style file names (`Input 1 - Task-8.pdf`). Expected in eval data;
  note it, but it does not undermine usability and does not block.
- **2** — One hard fabrication signal: an internal contradiction (a value, name,
  date, or address that conflicts with itself or another input), an impossible
  or implausible value, or content that could not have come from the claimed
  source. Material; usable only after a fix.
- **3** — Pervasive hard fabrication, or leaked content: the inputs contain
  pre-baked answers the agent is supposed to produce, leaked rubric or verifier
  IDs or task-pipeline markers, or contradictions extensive enough that the data
  cannot be trusted.

**Common hard-fail patterns.**
- Input files contain pre-baked analysis, totals, or conclusions that the agent
  is supposed to produce — i.e. the inputs leak the answer.
- Internal contradictions: the same entity carries two different values, names,
  dates, or addresses across or within files, or figures the task treats as
  authoritative do not reconcile.
- Impossible or implausible values: a closing time earlier than the opening
  time, a percentage outside 0–100, a date that cannot exist, a total that
  cannot come from its parts.
- Leaked task-pipeline artifacts aimed at the solver or grader and written into
  a document's CONTENT: rubric or verifier IDs, benchmark labels embedded in the
  body text (e.g. a clinical note that reads `(Input 1 and Input 2)`), bracketed
  generation markers, pre-baked answers, or instructions telling the agent what
  to do. (Data de-identification and public-dataset provenance are NOT
  task-pipeline artifacts. A benchmark-style *file name* alone — e.g.
  `Input 1 - Task-8.pdf` — is a cosmetic realism issue scored at severity 1, not
  a hard-fail; see Anti-overcall.)

**Anti-overcall.** Synthetic, anonymised, round, and templated data is expected
in benchmark and evaluation datasets and is not a defect by itself. Generic
names, round figures, clean formatting, and a consistent voice across documents
are soft markers: flag them at severity 1 at most, never as a reject, unless
they coincide with a hard signal above. Required synthetic identifiers
(placeholder SSNs, phone numbers, addresses in a tax or HR task) are mandated
for privacy and must not be scored as fabrication. **Legitimate document sign-off /
approval blocks** ("Prepared by / Reviewed by / Approved by [name, role]"),
review-cadence notes, and **`TBD` / `(to populate)` fields in an explicitly-draft or
incomplete document** (e.g. an unapproved budget) are AUTHENTIC business-document
artifacts — NOT "leaked reviewer notes" or fabrication markers. Reserve severity 2 and
above for a concrete, quotable hard signal. Quote the passage when flagging.

**De-identification and dataset provenance are expected obfuscation, not
defects.** Real professional data — healthcare/PHI especially, but also finance
and HR — is routinely de-identified: patient, provider, and staff names replaced
with bracketed redactions or role labels (`[Attending Physician]`,
`Day-Shift RN`), dates shifted, and records or images drawn from public
de-identified datasets. Provenance or attribution that names such a dataset or
its license (e.g. an NIH / Open-i chest-X-ray collection, a `CC BY-NC-ND` note,
a de-identified `UID` reference) is a legitimate sourcing artifact of obfuscated
eval data — score it severity 0, never as a reject. Do NOT treat de-identified
names, role substitutions, or dataset provenance as fabrication or as a
task-pipeline leak. The hard-fail above is reserved for markers aimed at the
solver or grader, not for obfuscation of real data.

**SVC decomposition-DAG exception (do NOT score as fabrication/leak).** These tasks form a
dependency DAG in which a node's input is often an *upstream node's completed databook*, provided
on purpose. Two patterns are BY DESIGN and legitimate: (1) **cross-reference / provenance notes**
that link a databook to its upstream source — e.g. `figures transcribe the verified L3 engine
(seed …)`, `ties to the L3-05B golden`, `Golden databook` / `Golden answer` provenance headers —
these are the intended upstream→downstream references the tasks rely on to locate downstream data,
not leaked task-pipeline markers; (2) **pre-computed figures in a provided upstream databook** the
instruction explicitly hands the agent as its input. Neither is a defect. Still flag genuine
fabrication: internal contradictions, impossible values, or leaked THIS-task rubric/verifier IDs,
bracketed generation markers, or reviewer notes. (3) **Returns/refunds as negative rows** — when the
orders/transactions export is described as including returns/refunds, negative-total rows (often with
credit refs like `c####`) are intended refund/credit contra-entries, NOT "impossible" values. Treat
the negative amount as by-design; at most a sev-1 note if the row's status fields are modeled
simplistically (e.g. a refund still marked paid/fulfilled).

**Benchmark-style file names are cosmetic, not a hard signal.** File names such
as `Input 1 - Task-8.pdf`, `Task-14-Input-2.png`, or `output-3.docx` are
trivially-renamable metadata, not document content. Score them at severity 1 at
most and never as a reject; only a benchmark label written into a document's
*body text* counts toward severity 2+.

---

### solution_instruction_alignment

```yaml
dimension_name: "Solution / Instruction Alignment"
check_kind: llm
inputs: [instruction, solution_files]
applies_when:
  builtin: solution_present
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Does the solution actually answer the task as posed?
Does it address each requirement, in roughly the right proportion?

**How to judge.** Extract the instruction's requirements (often a numbered or
bulleted list of things the deliverable must include). For each requirement,
locate where in the solution it is addressed and assess depth. A misalignment
can take two forms: missing a requirement entirely, or addressing it
superficially when the instruction asked for analysis.

**Severity rubric.**
- **0** — Addresses every stated requirement in proportion to its emphasis in
  the instruction.
- **1** — Addresses every requirement but with mild proportionality issues —
  one minor requirement underdeveloped, or one overemphasised at the expense
  of others.
- **2** — Misses or only superficially addresses one important requirement.
- **3** — Materially misses the point — multiple requirements unaddressed, or
  the deliverable answers a noticeably different question.

**Common hard-fail patterns.**
- Solution answers a noticeably different question than asked — the named
  subject, system, or person in the instruction is absent or peripheral in
  the deliverable.
- A required output element (named explicitly in the instruction) is missing
  entirely.
- The deliverable includes large unrequested sections (e.g. a "Methodology"
  or "Background" section the instruction never asked for) that crowd out
  the actual ask.
- The deliverable answers only part of a multi-part question (handles
  question 1 thoroughly, ignores or single-lines questions 2 and 3).

**Anti-overcall.** When the instruction prescribes a specific output
structure (named sections, ordered deliverables, format requirements),
the deliverable following that structure is *good alignment*, not
suspicious conformity. Only flag when the deliverable departs from what
was asked, not when it matches closely.

**SVC calibration.** Read ALL deliverables (databook + memo + deck), not just the model workbook,
before judging a requirement unmet. When the prompt says to treat a plan/figure as unapproved and
"choose the number you'd stand behind and show the downside," a golden that uses that figure as a
center is aligned IF it also (a) flags it unapproved and (b) recommends a sober alternative and
stresses the downside — check the memo/narrative for that treatment before concluding the model
"centers on the unapproved plan."

---

### verifier_rubric_clarity

```yaml
dimension_name: "Verifier Rubric Clarity"
check_kind: llm
inputs: [verifier_spec_json, instruction]
applies_when:
  builtin: task_has_verifier
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Across all verifier rubrics, are the criteria stated
clearly enough that two different judges would reach the same verdict on the
same solution?

**How to judge.** Read each rubric's `assertion.rubric.prompt` and ask
whether it would be interpreted the same way by two competent reviewers.
Watch for vague words ("appropriate", "reasonable", "sufficient") used
without anchoring, multi-clause rubrics that pack several distinct checks
into one, and rubrics whose pass/fail criterion is implicit rather than
stated.

**Severity rubric.**
- **0** — All rubrics state criteria unambiguously; two judges would agree.
- **1** — Most rubrics clear; one or two have minor ambiguity that wouldn't
  flip a verdict.
- **2** — Multiple rubrics are ambiguous in ways that would produce judge
  disagreement on borderline outputs.
- **3** — Pervasive ambiguity; the rubrics cannot reliably be applied without
  the author present.

**Common hard-fail patterns.**
- Multi-clause compound rubrics ("the memo includes findings AND
  recommendations AND a risk table") — partial credit is impossible because
  the whole rubric fails on any one piece.
- Pass condition stated as "no major issues" or "no obvious problems" — a
  default-pass rubric that any non-trivially-broken solution will pass.
- Use of "appropriate", "reasonable", "sufficient", "thorough", "robust" as
  pass conditions without any anchoring to observable features.
- Rubrics where the pass criterion is implicit — the rubric describes the
  topic to grade but never states what counts as passing.
- Rubrics that depend on undefined terms specific to the task without
  defining them.
- Advisory or optional phrasing on a check meant to be required — "should",
  "preferably", "ideally", "clearly shows", "as appropriate", or an example
  list introduced by "such as" / "e.g." that never states the required set. A
  judge cannot score a soft "should" as a hard pass/fail, so the same solution
  passes for one judge and fails for another.
- Side-effect / unsolicited-addition checks written with broad subjective
  language ("penalize any materially unsolicited additions") instead of an
  enumerated list of the specific forbidden artifacts (named extra files, tabs,
  sections). "Materially unsolicited" is not consistently gradable — enumerate
  what is disallowed.

**Not a clarity defect.** A soft quantifier ("substantially all", "most",
"approximately") is fine when the rubric also pins the concrete set or an
objective fallback — e.g. "substantially all 26 requirements appear by ID with
a coverage status and a net required quantity", where the enumerable IDs and
the fallback anchor make it gradable. Flag the soft word only when nothing
concrete backs it.

**SVC calibration.** SVC rubrics are `criteria[]`, not EKW `verifiers[]`: each criterion carries an
**`expected_value`** that anchors the pass condition. Do NOT flag a `description` as vague/implicit
when its `expected_value` supplies a concrete anchor (a number, band, or named item). Also, criteria
that spell out an explicit negation ("reporting the trough mean does NOT satisfy this"; "a P5 level
does NOT satisfy the drawdown") are **intentional disambiguation**, not compound defects — credit
them as clarity. Reserve severity for criteria genuinely un-anchored (no `expected_value`, no
observable pass condition).

---

### verifier_spec_parseable

```yaml
dimension_name: "Verifier Spec Parseable"
check_kind: programmatic
callback: check_verifier_parseable
inputs: [verifier_spec_json]
applies_when:
  builtin: task_has_rubric_verifier   # SVC: no verifiers[] array ⇒ skips (dropped on SVC)
gate:
  role: blocking
  reject_at_severity: 2          # an unparseable verifier is structurally broken
```

**What this evaluates.** Does `tests/verifier.json` parse cleanly against the
`VerifierSpec` Pydantic model used by the framework runtime?

**Callback behaviour.** Severity 0 if the spec parses; severity 2 if any field
is malformed or missing. The error message from the validation failure is
attached to the finding for reviewer drill-down.

---

### verifier_count_in_range

```yaml
dimension_name: "Verifier Count in Range"
check_kind: programmatic
callback: check_verifier_count_in_range
inputs: [verifier_spec_json]
applies_when:
  builtin: task_has_rubric_verifier   # SVC: no verifiers[] array ⇒ skips (dropped on SVC)
gate:
  role: advisory
  reject_at_severity: 3
config:
  min_verifiers: 6
  max_verifiers: null   # no upper limit — a thorough rubric with many checks is not a defect
```

**What this evaluates.** Are there enough verifier rubrics that coverage isn't
necessarily thin? This is a **minimum-only** gate: there is no upper limit on
verifier count.

**Callback behaviour.** Counts rubrics in the spec. Severity 0 if count is at or
above `min_verifiers`; severity 1 if short by 1–2; severity 2 if short by more.
When `max_verifiers` is null (the default), there is no penalty for a high count.

---

### verifier_outcome_grader_present

```yaml
dimension_name: "Verifier Outcome Grader Present"
check_kind: programmatic
callback: check_verifier_outcome_grader_present
inputs: [verifier_spec_json]
applies_when:
  builtin: task_has_rubric_verifier   # SVC: no verifiers[] array ⇒ skips (dropped on SVC)
gate:
  role: advisory
  reject_at_severity: 3
```

**What this evaluates.** Does the verifier include at least one
*outcome-grading* rubric? A verifier that grades only process or format
without checking the substance of the deliverable cannot drive a
meaningful reward.

**Callback behaviour.** Severity 0 when at least one rubric matches an
outcome marker; severity 2 otherwise. The check handles both verifier
spec schemas the framework encounters:

- Legacy GDPVal-216: `metadata.classification == "core"` (where "core"
  implied outcome-checking).
- Omega: `metadata.grading_role == "outcome"` (where importance and
  grading role are separate metadata fields).

A rubric matches if EITHER marker is set.

---
