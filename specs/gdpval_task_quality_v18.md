---
name: gdpval_task_quality
version: "18.6"
cache_bust: "9"

# ── Judge models ────────────────────────────────────────────────────────────
# Multi-judge ensemble. Disagreement between judges on a blocking dimension
# routes the sample to NEEDS_REVIEW rather than being silently majority-voted.
models:
  - model: gpt-5.5
    reasoning_effort: high
    max_tokens: 32000
  - model: gemini-3.1-pro-preview
    reasoning_effort: high
    max_tokens: 32000

# ── Per-sample fields surfaced by the loader ────────────────────────────────
# v18 dataset layout (one directory per task):
#   <task>/task.toml
#   <task>/instruction.md
#   <task>/environment/                      (agent inputs)
#   <task>/solution/golden_outputs/          (the canonical answer — graded)
#   <task>/tests/verifier.json               (the verifier definition — graded)
#   <task>/evals/                            (ignored by v18)
field_types:
  task_id: skip
  annotatorID: skip
  category: text
  difficulty: text
  tags: text
  instruction: text
  input_files: document
  solution_files: document          # canonical answer (one or more files)
  solution_reasoning: text          # optional Reasoning.docx narrative
  verifier_spec_json: text          # verbatim tests/verifier.json
  verifier_assertion_summary: text  # precomputed digest, optional
  intermodel_trajectory_digest: text  # condensed cross-model trajectories + per-rubric outcomes
  failed_trial_deliverable_digest: text  # lowest-reward trial's deliverable + per-rubric outcomes
  passed_trial_deliverable_digest: text  # highest-reward trial's deliverable + per-rubric outcomes

# ── Loader bindings — where each field comes from on disk ───────────────────
field_bindings:
  solution_files:
    source: directory
    path: solution/golden_outputs
    exclude: ["Reasoning.docx"]     # surfaced separately as solution_reasoning
  solution_reasoning:
    source: file
    path: solution/golden_outputs/Reasoning.docx
    optional: true
  verifier_spec_json:
    source: file
    path: tests/verifier.json
    optional: true
  verifier_assertion_summary:
    source: callback
    callback: build_verifier_assertion_summary
    optional: true
  intermodel_trajectory_digest:
    source: callback
    callback: build_intermodel_trajectory_digest
    optional: true
  failed_trial_deliverable_digest:
    source: callback
    callback: build_failed_trial_deliverable_digest
    optional: true
  passed_trial_deliverable_digest:
    source: callback
    callback: build_passed_trial_deliverable_digest
    optional: true

# ── Synthesis layer ─────────────────────────────────────────────────────────
# After all dimensions run, the synthesizer makes ONE LLM call per sample to
# produce the SampleReview: a verdict, a one-line summary, top-K ranked
# actionable issues, and a suggested-fixes list. This is the artifact shown
# to reviewers; raw per-dim judgements are retained for audit only.
synthesis:
  enabled: true
  top_issues_cap: 3
  judge_model: gpt-5.5
  judge_reasoning_effort: high
  judge_max_tokens: 16000

# ── Calibration ─────────────────────────────────────────────────────────────
# A small directory of (sample, expected_verdict) pairs the framework runs
# against this spec on every build. Drift between expected and observed
# verdicts is reported in batch_summary.md. Populated incrementally as we
# accumulate confirmed cases.
calibration:
  goldens_path: calibration/v18/
  enabled: true

# ── Severity scale (referenced by every LLM dimension's rubric) ─────────────
# The scale is GLOBAL. Most dimensions inherit "sev 2 = fixable, sev 3 =
# structural" semantics, but a few dimensions (overall_gdpval_fit,
# instruction_input_consistency, professional_complexity, the two authenticity
# dims, and verifier_spec_parseable) are structural by their nature — even
# their sev-2 findings cannot be fixed by editing the sample. Those dimensions
# carry `reject_at_severity: 2`. Spec authors adding new dimensions should
# pick `reject_at_severity` based on whether sev 2 for that dimension is
# fixable or structural; the rubric severity levels themselves should always
# follow the global scale.
severity_scale:
  0: "no issue or fully meets criterion"
  1: "minor concern; does not affect usability of the sample"
  2: "material concern; sample is usable only after a fix"
  3: "severe failure; sample cannot be made usable by a fix"

# ── Evidence policy ─────────────────────────────────────────────────────────
# Every LLM judge MUST quote a verbatim passage from the inputs to justify
# any finding with severity >= 1. Findings without an evidence quote are
# programmatically demoted to severity 0. This is the principal anti-overcall
# guard.
require_evidence_quote: true

# ── Confidence policy ───────────────────────────────────────────────────────
# Judges return a confidence ∈ {low, medium, high}. Low-confidence severity-1
# findings are demoted to 0; low-confidence severity-2+ findings trigger
# NEEDS_REVIEW for the sample rather than autoacting on them.
confidence_demotion:
  low_at_sev1: demote_to_0
  low_at_sev2_or_higher: route_to_review

# ── Verdict tiering (used by the synthesizer) ───────────────────────────────
# Five-tier verdict computed from all dimensions' findings + gating rules:
#
#   REJECT_STRUCTURAL    any blocking dim at sev >= reject_at_severity with
#                        fixability=structural
#   REJECT_FIXABLE       any blocking dim at sev >= reject_at_severity with
#                        fixability != structural   (rare — most reject-worthy
#                        issues will be fixable, but explicit tier exists)
#   NEEDS_REVIEW         judge disagreement on a blocking dim, OR a
#                        low-confidence finding at sev >= 2, OR finding lands
#                        exactly at reject_at_severity - 1 on a blocking dim
#                        (borderline)
#   ACCEPT_WITH_FIXES    only advisory issues with severity >= 1, OR blocking
#                        issues at severity < reject_at_severity
#   ACCEPT               no findings of severity >= 1 on any blocking or
#                        advisory dim (informational findings allowed)
verdict_tiers:
  - reject_structural
  - reject_fixable
  - needs_review
  - accept_with_fixes
  - accept
---

# GDPVal Task Quality Spec — v18

This spec defines how the quality framework grades GDPVal-style data samples for v18.
It is a **clean break** from v17 — it does not inherit from any prior spec. The unit
of evaluation is a single task directory containing an instruction, a set of input
files, a canonical "golden" solution under `solution/golden_outputs/`, and a verifier
definition under `tests/verifier.json`. v18 does **not** evaluate model outputs.

Each dimension below declares:

- **inputs** — which per-sample fields the judge sees
- **gate** — how findings translate into the sample's verdict (`role`,
  `reject_at_severity`, `on_disagreement`)
- **applies_when** — optional precondition; the dimension is skipped if false
- **severity rubric** — anchored definitions for severity levels 0–3
- **common hard-fail patterns** — concrete diagnostic patterns to flag with
  evidence; judges decide final severity in context, not by auto-jump

Five families of dimensions ship in v18: **Task Structure**, **Authenticity**,
**Solution Quality**, **Verifier Quality** (conditional), and **Programmatic
Checks**.

**Note on gating.** The default `reject_at_severity` is `3`, which means a finding
must reach the unfixable tier to auto-reject. Six dimensions tighten this to `2`
because their sev-2 findings are *also* unfixable in practice — see the per-dim
gate blocks for the rationale.

**Note on hard-fail patterns.** Several dimensions list concrete diagnostic
patterns under "Common hard-fail patterns". These are *guidance for the judge*,
not automatic severity bumps. When a judge sees one of these patterns clearly
present, they should quote the evidence and decide the severity in context.
A pattern that's borderline, ambiguous, or has a plausible benign explanation
should not be flagged.

**Important: evaluation-context conventions.** This corpus is an *evaluation
dataset*, not a copy of in-the-wild work. Several patterns that would be
unusual in a forwarded email are perfectly reasonable as eval setup, and
**must not** be penalised as evidence of being synthetic or constructed:

- **Persona / role assignment.** Phrases like "You are a senior credit
  analyst at a direct-lending fund" or "Acting as the portfolio manager,
  …" set the agent's context. They are not "LLM-shaped framing."
- **Output file specifications.** Instructions that say "save your memo as
  `/workspace/output/memo.docx`" or "produce a `.xlsx` model with sheets
  named …" are eval-infrastructure necessities: the verifier needs to know
  where to look. Naming the deliverable, its file format, or its location
  is not "pre-specifying the structure" in the synthetic sense.
- **Required deliverable list.** An instruction that says "produce two
  files: a recommendation memo and a supporting model" is fine. The
  synthetic version is an instruction whose enumerated requirements read
  as a verifier rubric copied verbatim into the prompt (e.g. "include
  section 1: …, section 2: …, section 3: …" mirroring the rubric items).
- **Length / scope cues.** "A one-page memo" or "a ~500-word summary" is
  a normal professional ask. Only flag length constraints that look
  benchmark-arbitrary (e.g. "exactly 3 bullet points", "exactly 200 words").
- **Capability framing.** Eval tasks are *designed* to exercise model
  capabilities; that is what an eval is. Only flag tasks whose stakes or
  domain are so paper-thin that they couldn't plausibly represent any
  real delegated work.
- **Acceptance criteria in the instruction.** Real professional briefs
  sometimes include "the deliverable must address …" lists. Fine.
- **Anonymisation conventions.** Redacted-real or anonymised real
  documents may carry generic-sounding names without being fabricated.

The anti-overcall guidance on each dimension reflects these conventions.
Judges should err toward "the eval needed this" rather than "this is
synthetic" when a pattern could plausibly fall in either bucket.

---

## Task Structure

### overall_gdpval_fit

```yaml
dimension_name: "Overall GDPVal Fit"
check_kind: llm
inputs: [instruction, category, solution_files]
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

---

### instruction_input_consistency

```yaml
dimension_name: "Instruction / Input Consistency"
check_kind: llm
inputs: [instruction, input_files]
gate:
  role: blocking
  reject_at_severity: 2          # sev 2 = "input missing/mismatched"; can't fix by editing the sample
  on_disagreement: review
```

**What this evaluates.** Does the instruction match the inputs actually provided?
A solving agent should be able to read the instruction and locate every file,
section, or data element the instruction references — and the inputs should
contain the substance the instruction relies on.

**How to judge.** Walk the instruction looking for explicit references (file
names, section names, "the attached X", named data tables) and verify each
exists in `input_files`. Then check the inverse: are there inputs that the
instruction doesn't reference but obviously expects the agent to find? Finally,
spot-check that the inputs contain real substance, not just placeholder data.

**Severity rubric.**
- **0** — Instruction and inputs are fully consistent. Every reference resolves;
  no important input goes unmentioned.
- **1** — Minor mismatch (filename has a small typo; instruction says "the
  spreadsheet" and the file has a slightly different name). The agent can
  reasonably infer what was meant.
- **2** — A relied-upon input is missing or mismatched in a way that forces the
  agent to guess what data to use. An attached file is referenced but not
  provided; or the data table the instruction needs is absent.
- **3** — The instruction references inputs that don't exist at all, or
  contradicts the inputs that do (numbers disagree, dates contradict, roles
  conflict).

**Common hard-fail patterns.**
- Instruction explicitly names a file by filename that is not present in
  `input_files`.
- A specific number, date, or named entity in the instruction directly
  contradicts the same item in the inputs (e.g. instruction says "the Q3
  report", inputs are dated Q2).
- Instruction references sections, tables, or appendices ("see Appendix B")
  that don't exist in the named input file.
- Instruction and inputs use incompatible scope (instruction asks about
  "all three regions", inputs cover only two).

---

### input_prompt_leakage

```yaml
dimension_name: "Input Folder Prompt Leakage"
check_kind: llm
inputs: [instruction, input_files]
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

---

### self_containedness

```yaml
dimension_name: "Self-Containedness"
check_kind: llm
inputs: [instruction, input_files]
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Can the task be solved from the given instruction and
inputs alone, or does it implicitly require things the agent doesn't have —
external context, credentials, network access, or undocumented domain knowledge?

**How to judge.** Read the instruction as if you were the agent. Ask: do I have
everything I need to begin? If the task needs external lookup (current stock
price, today's date, a specific company filing), is that information embedded
in the inputs, or does the agent have to fetch it? Tools the agent already has
(Python, document libraries, basic web search if granted) do not count as
external dependencies; the question is about *information* and *context*, not
runtime capabilities.

**Severity rubric.**
- **0** — Fully self-contained. Instruction + inputs supply everything the agent
  needs.
- **1** — Requires general professional knowledge any competent agent would have
  (basic accounting conventions, common formatting expectations). Not a defect.
- **2** — Requires undocumented domain context — specific company policies,
  internal jargon defined nowhere, conventions the agent couldn't reasonably
  guess.
- **3** — Cannot be solved without information that isn't provided at all
  (external credentials, a specific filing not attached, results of a
  proprietary system).

**Common hard-fail patterns.**
- Instruction requires real-time external data not embedded in inputs (today's
  date, current stock price, live system state, recent news).
- Instruction references proprietary company policies, internal documents, or
  named systems with no documentation supplied.
- Instruction implies credentials or login to external resources are needed.
- "As discussed in our last meeting…" / "Per our previous email…" framings
  with no record of the prior interaction provided.
- Task requires consulting a specific named external party ("ask Legal", "check
  with Finance") with no fallback context.

**Anti-overcall.** Many tasks legitimately require *general* professional
knowledge (e.g. "write a memo to the CFO" doesn't need to define what a CFO is).
Only flag missing context that is *specific* and *not inferrable*.

---

### role_and_deliverable_clarity

```yaml
dimension_name: "Role and Deliverable Clarity"
check_kind: llm
inputs: [instruction]
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Two separate clarities, both needed: **role** (who is
the agent supposed to be — analyst, attorney, examiner) and **deliverable**
(what concrete artifact must be produced, in what format).

**How to judge.** Role can be stated explicitly ("as a senior credit analyst")
or inferable from the framing of the request. Deliverable should specify the
artifact type ("a memo", "a model", "a one-page summary") and ideally the file
format. Either being ambiguous is a problem; both being ambiguous compounds it.

**Severity rubric.**
- **0** — Both role and deliverable are crisp and unambiguous; a competent agent
  knows immediately what shape the answer should take.
- **1** — Minor ambiguity in one of the two — usually the deliverable's file
  format is unstated but reasonably inferable.
- **2** — Material ambiguity in either dimension. Reasonable agents would
  produce noticeably different shapes of deliverable.
- **3** — Either role or deliverable is essentially undefined, or the two are
  contradictory.

**Common hard-fail patterns.**
- Instruction assigns conflicting roles ("you are a junior analyst, produce a
  board-level strategic recommendation").
- Deliverable type is never stated and cannot be inferred from context (no
  hint of memo vs. report vs. spreadsheet vs. presentation).
- Multiple deliverables requested but the count is ambiguous ("produce
  documentation" could mean one file or many; "supporting materials" plural
  but unclear how many).
- "Write something useful" / "produce a deliverable" / other vague asks with
  no shape.

---

### professional_complexity

```yaml
dimension_name: "Professional Complexity"
check_kind: llm
inputs: [instruction, input_files, solution_files]
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

---

### feasibility_and_clarity

```yaml
dimension_name: "Feasibility and Clarity"
check_kind: llm
inputs: [instruction, input_files]
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Two related questions: **feasibility** (is the task
doable within the stated scope and given inputs?) and **interpretive clarity**
(would two competent professionals interpret the request the same way?).

**How to judge.** This dim is distinct from `instruction_input_consistency`
(which is about whether referenced inputs exist) and from
`role_and_deliverable_clarity` (which is about who and what). Here, ask: given
the inputs, is the task reasonably scoped to be completed in the implied time?
And: if you asked two senior practitioners to do this, would their deliverables
have the same shape?

**Severity rubric.**
- **0** — Clearly feasible; instruction would be interpreted identically by any
  competent professional.
- **1** — Feasible with mild ambiguity that wouldn't change the deliverable's
  shape (e.g. exact memo length unstated but obvious from context).
- **2** — Feasibility is doubtful (too much work for the implied scope, or too
  little information) *or* the instruction is ambiguous enough to produce
  noticeably different answers.
- **3** — Infeasible within the stated scope, or fundamentally unclear.

**Common hard-fail patterns.**
- Instruction asks for analysis the inputs don't actually contain the data to
  support (e.g. "compare revenue trends across 5 years" with one year of data).
- Requested deliverable scope is implausible for the implied timeframe
  ("comprehensive 50-page report in two hours").
- Two stated requirements directly contradict each other (the deliverable
  cannot satisfy both simultaneously).
- The instruction has two plausible interpretations that would lead to
  materially different deliverables.

**Anti-overcall.** Eval-context conventions (named deliverables, file
paths, length cues, persona prompts) are not interpretive ambiguity. When
the instruction specifies an output format or location, that *removes*
ambiguity rather than creating it.

---

### metadata_completeness

```yaml
dimension_name: "Metadata Completeness"
check_kind: programmatic
callback: check_metadata_completeness
inputs: [category]
gate:
  role: informational
  reject_at_severity: 3
```

**What this evaluates.** Does `task.toml` carry the minimum metadata the
framework relies on — specifically, a non-empty `category` field?

**Callback behaviour.** The callback returns severity 0 if `category` is
present and non-empty; severity 2 otherwise. v18 deliberately treats most
metadata as informational — missing tags or difficulty values surface in the
batch summary but do not affect the verdict.

---

## Authenticity

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
  Markdown.
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
for privacy and must not be scored as fabrication. Reserve severity 2 and above
for a concrete, quotable hard signal. Quote the passage when flagging.

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

**Benchmark-style file names are cosmetic, not a hard signal.** File names such
as `Input 1 - Task-8.pdf`, `Task-14-Input-2.png`, or `output-3.docx` are
trivially-renamable metadata, not document content. Score them at severity 1 at
most and never as a reject; only a benchmark label written into a document's
*body text* counts toward severity 2+.

---

## Solution Quality

These dimensions evaluate the contents of `solution/golden_outputs/` as the
canonical answer to the task. They do not compare the solution to any model
output and do not treat the solution as "an improved version" of anything — it
is simply the golden answer being assessed on its own terms.

### solution_format_match

```yaml
dimension_name: "Solution Format Match"
check_kind: llm
inputs: [instruction, solution_files]
applies_when:
  builtin: solution_present
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Does the solution file (or set of files) match the
deliverable format the instruction asks for? File types, expected file names
if specified, expected number of deliverables.

**How to judge.** Extract from the instruction what the deliverable format
should be — "memo" → .docx, "model" → .xlsx, "summary" → either, "report" →
.docx or .pdf — and verify the solution files match. If the instruction
specifies file names, check those too. If multiple deliverables are
requested, confirm all are present.

**CRITICAL: judge by the ORIGINAL file extension, not the transport
format.** Each `<file>` tag exposes both the original `name` (e.g.
`memo.docx`) and an `original_extension` attribute (e.g. `.docx`). The
framework converts Office files to PDF for multimodal transport to the
judge, so a `.docx` deliverable may appear as a PDF document block with
`original_extension=".docx"` and `sent_to_judge_as="pdf-conversion-of-original"`.
The deliverable's ACTUAL format is `.docx` in that case — do NOT flag
this as a format mismatch.

**Severity rubric.**
- **0** — Exact format match; all expected deliverables present in the
  expected formats.
- **1** — Minor format mismatch (extra supporting file, slight naming
  variation, deliverable in a related format).
- **2** — Material format mismatch — one deliverable is in the wrong type, or
  a required deliverable is missing while others are present.
- **3** — Fundamentally wrong format (instruction asks for a spreadsheet, all
  files are memos; or no recognisable deliverable exists).

**Common hard-fail patterns.**
- File extension doesn't match the requested deliverable type (e.g. `.txt`
  when a `.docx` memo was requested, `.md` when a polished document was
  requested).
- Wrong number of files (a single `.docx` when memo + supporting spreadsheet
  were both requested).
- A solution file is empty or near-empty when the request implied a
  substantive deliverable.
- File name explicitly contradicts a name the instruction specified
  (instruction asks for `q3_analysis.xlsx`, solution provides
  `Analysis.xlsx`).

---

### solution_substantiveness

```yaml
dimension_name: "Solution Substantiveness"
check_kind: llm
inputs: [solution_files, solution_reasoning]
applies_when:
  builtin: solution_present
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Is the solution substantive professional work — real
analysis, real structure, real conclusions — or is it a skeleton, stub, or
placeholder-heavy draft?

**How to judge.** Read the actual content of each solution file. Substantive
work has specific numerical results, named entities, defensible conclusions,
and developed arguments. Skeleton work has section headings followed by
one-liners, placeholder values, "TBD" markers, or generic filler. The optional
`solution_reasoning` (Reasoning.docx) can help calibrate intent but the
substantive content must be in the deliverable files themselves.

**Redacted identity fields are not placeholders.** In de-identified professional
deliverables (healthcare/PHI especially), a preparer, patient, or provider name
shown as a bracketed redaction or a role label (`Prepared by: Day-Shift RN`,
`[Provider Name]`, `[RN Name]`) is expected de-identification / obfuscation, not
skeleton filler — do not score it as a substantiveness defect and do not treat a
role-instead-of-name as a "missing name." The placeholder hard-fail below targets
missing CONTENT — absent analysis, `[INSERT ANALYSIS]`, `TBD` sections — not
identity redaction.

**Severity rubric.**
- **0** — Fully substantive across all deliverable files; demonstrates real
  analytical effort.
- **1** — Mostly substantive with one or two thinly-developed sections.
- **2** — Materially incomplete; large sections are placeholders, one-liners,
  or summaries of what should go there.
- **3** — Skeleton-only or pervasive use of "[INSERT X]" / "TBD" / template
  filler patterns; the deliverable is not usable as-is.

**Common hard-fail patterns.**
- LLM closing phrases pasted into formal deliverable documents where they
  don't belong: "I hope this helps" / "Let me know if you need anything
  else" / "Feel free to reach out" / "Based on the analysis above…" appearing
  at the end of a *report* or *memo body*. (These same phrases at the close
  of an actual professional email are fine — judge by document type.)
- LLM-characteristic hedging in formal documents: "It's important to note
  that…", "It's worth highlighting that…", "It should be noted…" used
  pervasively as filler. Occasional use in real documents is fine.
- Citations to fabricated sources — named references like "(Smith 2023)" or
  footnotes to documents that don't appear in the inputs and look invented.
- Section headers followed by content less than two lines long, or section
  headers followed by content that only restates the heading.
- "[INSERT X]" / "[TBD]" / "[Placeholder]" / "<your data here>" content
  placeholders anywhere in the deliverable. (Identity redactions such as
  `[Patient Name]` or a role in a signature/"Prepared by" line are
  de-identification, not content placeholders — exclude them.)
- Conclusions or recommendations that don't follow from the body — the
  recommendation contradicts the analysis, or appears with no supporting
  argument.
- Generic bullet lists with no task-specific content ("• Plan", "• Execute",
  "• Review") used as substantive content.

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

---

### solution_internal_consistency

```yaml
dimension_name: "Solution Internal Consistency"
check_kind: llm
inputs: [solution_files]
applies_when:
  builtin: solution_present
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Is the deliverable internally consistent — do the
numbers reconcile, do the conclusions follow from the analysis, do the
deliverable files agree with each other where they overlap?

**How to judge.** When the solution contains multiple files (e.g. a memo plus
a spreadsheet, or a report plus a presentation), look for cross-references
and verify they agree. Within a single file, check that key numbers cited
in the narrative match the numbers in tables/charts, that percentages sum
where they should, and that the conclusion is supported by the body. This dim
is distinct from `solution_substantiveness` (which asks "is there enough
there") and `solution_instruction_alignment` (which asks "does it answer the
question"); here we ask "does it agree with itself".

**Severity rubric.**
- **0** — Fully internally consistent. Numbers reconcile, conclusions follow,
  files agree with each other.
- **1** — One minor inconsistency (a number rounded differently in two places;
  a small percentage-sum issue that doesn't change the conclusion).
- **2** — A material inconsistency — a key number disagrees between memo and
  model, or a stated conclusion is contradicted by the body, or two
  deliverable files give different answers to the same question.
- **3** — Pervasive or severe inconsistencies that fundamentally undermine the
  deliverable's credibility.

**Common hard-fail patterns.**
- Numbers cited in the narrative don't match numbers in tables or attached
  spreadsheets (e.g. memo says "revenue grew 12%", model shows 8%).
- Percentages that should sum to 100% don't sum to 100% (allocation tables,
  category breakdowns).
- The executive summary's bottom line contradicts the body analysis (summary
  says "recommend approval", body argues for further study).
- Cross-file inconsistency — memo references a value the spreadsheet doesn't
  contain, or two files give different totals for the same item.
- Date inconsistencies — different dates given for the same event in
  different parts of the deliverable.
- Internal arithmetic that doesn't check (line items in a table don't sum to
  the stated total; subtotals are wrong).

---

### solution_layout_professionalism

```yaml
dimension_name: "Solution Layout Professionalism"
check_kind: llm
inputs: [solution_files]
applies_when:
  builtin: solution_present
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Is the solution presented to professional standard —
appropriate document structure, headings, formatting, length, no obvious
artifacts of careless drafting?

**How to judge.** Look at document-level concerns: heading hierarchy,
consistent formatting, page breaks where expected, tables that render cleanly,
charts that are titled and legend-equipped. This dim is about *presentation*,
not analytical depth (covered by substantiveness) or correctness (covered by
alignment).

**Severity rubric.**
- **0** — Polished and professional throughout; ready to send.
- **1** — Generally good with one or two formatting rough edges (an
  inconsistent heading style, a slightly misaligned table).
- **2** — Multiple meaningful formatting issues (broken layout, unclear
  hierarchy, charts without titles, awkward page breaks).
- **3** — Reads as a rough draft; significant cleanup needed before it could
  be sent to a real audience.

**Common hard-fail patterns.**
- Markdown formatting visible as raw text in a non-markdown deliverable —
  `**bold**` shown literally, `-` characters rendering as plain text instead
  of bullets, leading `#` characters as headings in a `.docx`.
- Inconsistent heading hierarchy (H1 followed by H3 followed by H2, or
  visually-similar headings used for differently-leveled content).
- Mixed fonts or sizes across sections that should be uniform.
- Tables overflow page boundaries or have visibly misaligned columns.
- Charts pasted as low-resolution raster screenshots in a document where
  editable charts are expected.

---

### solution_chart_quality

```yaml
dimension_name: "Solution Chart Quality"
check_kind: llm
inputs: [solution_files]
applies_when:
  builtin: solution_has_charts
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** If the solution contains charts/visualisations, are
they readable, appropriately labelled, and visually correct?

**Skip condition.** This dimension is skipped (no LLM call made) for any
sample where the solution contains no chart-like content. The framework
detects "chart-like" programmatically — embedded images in .docx, chart
sheets in .xlsx, vector chart objects in .pdf.

**How to judge.** When charts are present, examine: axis labels and units,
chart titles, legends, colour/contrast choices, data-ink ratio, and basic
visual accuracy (does the chart correctly represent the underlying data).

**Severity rubric.**
- **0** — Charts are clean, well-labelled, and visually accurate.
- **1** — Minor chart issues (axis label too small, legend placement, one
  questionable colour choice).
- **2** — Material chart problems (mis-scaled axes, unclear or missing labels,
  ambiguous categories).
- **3** — Charts are misleading or unreadable.

**Common hard-fail patterns.**
- Charts with no title or no axis labels at all.
- Y-axis truncated in a way that misleads (e.g. starting at 90% to exaggerate
  a small variation).
- 3D pie or bar charts with no analytical reason for the third dimension
  (decorative 3D distorts perception).
- Pie charts with more than 5 slices (becomes unreadable).
- Two or more charts on the same page showing essentially the same data in
  slightly different forms.
- Charts pasted at low resolution / visibly image-compressed.

---

### solution_visual_insight_surfacing

```yaml
dimension_name: "Solution Visual Insight Surfacing"
check_kind: llm
inputs: [instruction, solution_files]
applies_when:
  builtin: solution_has_charts
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Do the solution's charts actually surface the
analytical insight the task is asking about, or are they decorative?

**Skip condition.** Same as `solution_chart_quality` — skipped if no charts.

**How to judge.** Identify the key analytical question the task poses, then
ask whether the charts help the reader see the answer. A bar chart of raw
inputs is decorative; a bar chart highlighting the three findings that drove
the recommendation surfaces insight. This dim is distinct from
`solution_chart_quality` (which is about chart correctness, not utility).

**Severity rubric.**
- **0** — Charts pull out and highlight the key insight effectively.
- **1** — Charts show the underlying data but the insight requires some
  interpretation by the reader.
- **2** — Charts are present but largely decorative; the actual analytical
  insight lives in the text.
- **3** — Charts mislead or distract from the task's actual analytical
  content.

**Common hard-fail patterns.**
- Charts visualise only raw input data and never any derived/summary view —
  the analytical work is in the text, the charts just decorate.
- Multiple charts that all answer the same analytical question in slightly
  different forms, with no progression of insight.
- Charts that appear in the deliverable but are never referenced or
  discussed in the narrative.
- Charts whose visual emphasis (size, colour, position) misallocates
  attention relative to the analytical priorities.

---

## Verifier Quality

These dimensions evaluate `tests/verifier.json` as a definition document —
how well it captures the criteria a correct solution must meet. They are
**conditional** on the task having a verifier spec; the framework skips them
entirely if `tests/verifier.json` is missing.

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

---

### verifier_rubric_specificity

```yaml
dimension_name: "Verifier Rubric Specificity"
check_kind: llm
inputs: [verifier_spec_json, instruction]
applies_when:
  builtin: task_has_verifier
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Are the rubrics concrete enough to grade against, or
do they ask vague qualitative questions that require the judge to invent the
criteria?

**How to judge.** A specific rubric asks something observable: "the document
includes a synthesis table of the three audit findings cross-mapped to
federal requirements". A non-specific rubric asks something interpretive:
"the analysis is high quality". The distinction is whether the rubric points
at concrete features of the solution.

**Severity rubric.**
- **0** — All rubrics state concrete, observable criteria.
- **1** — Most rubrics specific; one or two lean on vague qualitative
  language but are mostly anchored.
- **2** — Several rubrics rely on undefined qualitative judgments
  ("appropriate", "professional", "thorough" without anchoring).
- **3** — Rubrics are predominantly impressionistic; the judge would have to
  invent most of the criteria.

**Common hard-fail patterns.**
- Rubric asks the judge to perform substantive analysis themselves to grade
  ("the financial model correctly projects revenue", "the recommendation is
  optimal") — the rubric requires the judge to redo the work.
- Subjective qualitative judgments without observable criteria ("the analysis
  is thorough", "the prose is professional", "the tone is appropriate").
- Rubric criteria depend on undefined task-specific terms ("ensure the
  appropriate tone for the audience" with no anchor for "appropriate").
- Rubric grades aesthetic preferences rather than functional correctness
  ("the document is well-organised" without specifying what organisation
  means here).

---

### verifier_rubric_construct_validity

```yaml
dimension_name: "Verifier Rubric Construct Validity"
check_kind: llm
inputs: [verifier_spec_json, instruction, solution_files, verifier_assertion_summary, failed_trial_deliverable_digest]
applies_when:
  builtin: task_has_verifier
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Do the rubrics actually measure the quality they
claim to measure? If a rubric is named "instruction alignment" or framed as
checking output quality, does it in fact distinguish aligned outputs from
misaligned ones — or does it measure a proxy that happens to correlate?

**How to judge.** For each rubric, identify what it *claims* to assess (often
clear from its name + how_justification + why_justification metadata), then
examine whether the assertion text actually tests that. A common failure is
rubrics that check surface features (presence of a heading named "Findings")
while claiming to grade substance (whether findings are actually present).

When `verifier_assertion_summary` is provided, also use the actual pass/fail
patterns it reports. A rubric that *claims* to measure substance but
appears in the "always-passes" list of the summary is a strong construct-
validity red flag — any plausible attempt passed it, so it isn't actually
measuring what it claims.

**A 0% (always-fails) pass rate is not by itself a construct defect.** Before
flagging a rubric that no evaluated model passes, check whether the golden
solution in `solution_files` satisfies it. If the golden passes, the low pass
rate means the task is hard — the models hit the intended break — not that the
rubric is broken or proxy-grading; leave it at severity 0. Treat an
always-fails rubric as a construct defect only when the golden itself would
fail it, or when the requirement it encodes is not grounded in the instruction
or a stated policy.

**Severity rubric.**
- **0** — Every rubric is a valid construct for what it claims to measure.
- **1** — Most rubrics valid; one or two measure adjacent properties rather
  than the stated target.
- **2** — Multiple rubrics measure proxies that don't reliably indicate the
  intended construct.
- **3** — Rubrics measure something materially different from what they
  claim, or rely on proxies so loose they don't discriminate.

**Common hard-fail patterns.**
- Rubric checks the expert solution's specific section ordering, headings,
  or phrasing rather than requirements stated in the instruction.
- Rubric checks surface presence ("includes a section titled 'Findings'")
  while its justification claims it grades substance ("the findings are
  comprehensive").
- Rubric tests alphabetical or positional conventions the instruction never
  specified (e.g. "states are listed alphabetically" when ordering wasn't
  required).
- Rubric tests for the expert's specific stylistic choices (paragraph length,
  tone, bullet vs. prose) that weren't required by the task.
- Rubric claims to verify live / formula-driven values, auditability, or "no
  hardcoding" but its source is `extract_text`. Extracted text shows only the
  rendered number, so the rubric cannot confirm a value is formula-driven or
  untyped — it grades a proxy (the number is present) while claiming to grade
  traceability. The fix is concrete proxies the extracted text CAN confirm:
  required intermediate tabs populated, numeric tie-outs between tabs, key
  anchor cells present.

**Not a hard fail (eval-context convention).** Verifiers that demand
specific numerical or factual values (e.g., "the memo outputs $1.24M as
the final figure", "the model projects 8.7% growth") are LEGITIMATE
deterministic outcome checks for tasks where there is a uniquely-correct
answer. Do NOT flag value-specific verifiers as a construct-validity
problem — they are doing exactly what deterministic verification should
do. Flag them only when the specific value being checked clearly doesn't
correspond to a uniquely-correct answer (e.g. a value that varies with
defensible interpretation choices).

Likewise, requiring the deliverable to **state prompt-grounded facts** (figures,
totals, counts the instruction asks for) is a completeness check, not
over-penalization — even if no evaluated model happened to include them. Do not
flag such a rubric as too strict as long as it accepts the fact in any form
(prose or table) and does not demand a specific layout the instruction never
required. Flag format-rigidity only when the rubric fails a correct answer for
its FORMAT (e.g. requires a table when prose conveys the same fact).

**Before calling a rubric over-strict, check the golden.** A rubric that a
*correct* solution can satisfy is not over-penalizing — it is merely demanding.
The golden solution in `solution_files` is that reference: if the golden passes
the rubric, do not flag it as too strict or over-specified, even if the
requirement feels heavy or un-asked-for. Flag over-penalization only when the
rubric would fail the golden (or another defensible correct answer) — i.e. it
rejects a right answer, not just a lazy one.

**Use the failed-trial exemplar as concrete over-strictness evidence.** When
`failed_trial_deliverable_digest` is provided, it contains the ACTUAL deliverable
text of the lowest-scoring non-crash model run plus that trial's per-rubric
outcomes and the judge's justification for each failure. This is your strongest
signal for over-strictness: read the deliverable, then read why each core rubric
failed it. If the deliverable is a defensible, on-task answer that a rubric
rejected for an unstated formatting/wording/detail requirement — especially when
the failure justification itself concedes the substance is correct ("matches the
required order … however does not explicitly state …") — that rubric is
over-strict (construct-invalid). If instead the deliverable is genuinely wrong or
missing required content, the failure is legitimate — do not flag. This exemplar
supersedes the always-fails digest signal: a concrete right-answer-marked-wrong
outranks an aggregate pass-rate. Weigh it against the golden per the rule above.

---

### verifier_outcome_coverage

```yaml
dimension_name: "Verifier Outcome Coverage"
check_kind: llm
inputs: [verifier_spec_json, instruction, solution_files, verifier_assertion_summary]
applies_when:
  builtin: task_has_verifier
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Do the rubrics together cover the important *outcomes*
of a correct solution? "Outcome" means "did the solution achieve the intended
result", as opposed to "process" rubrics that only check whether expected
steps were taken.

**How to judge.** Enumerate the substantive outcomes a good solution must
achieve (drawn from the instruction). For each, check that at least one
rubric in the verifier graders that outcome. Coverage failures are typically
verifiers that check format/structure thoroughly but leave the substantive
correctness ungraded.

When `verifier_assertion_summary` is provided, use it as a second lens.
Rubrics in the "always-passes" list are weak coverage — they pass for any
plausible attempt, so they don't actually contribute to outcome
discrimination. If many rubrics are in that list, the verifier likely has
broad nominal coverage but thin effective coverage.

**Severity rubric.**
- **0** — All major outcome aspects are covered by at least one rubric.
- **1** — Coverage is broad with one minor gap.
- **2** — A material outcome aspect (something the task is centrally about)
  is uncovered.
- **3** — Coverage is fundamentally lacking — multiple critical outcomes
  ungraded; the verifier could pass a wrong solution.

**Severity reflects under-coverage risk.** Reserve severity 2+ (which gates) for
gaps where a materially wrong solution would pass because an outcome is
ungraded. Double-counting and other over-weighting problems — the same failure
penalised by two rubrics — distort the reward's ranking but do not let a wrong
solution through; treat them as at most severity 1 unless the distortion is
heavy enough to change the overall verdict. A missing check on SIDE-EFFECTS or
output hygiene — no rubric lists the output directory to catch forbidden extra
files, or unsolicited additions go ungraded — is likewise a completeness gap on
secondary behavior, not on the core outcome: a solution that produces the
correct deliverable plus an extra file is still substantively correct, so treat
it as at most severity 1 (accept-with-fixes polish). Reserve severity 2+ for the
core required outcomes being ungraded (including a guardrail-only rubric that
never grades the positive quality of the required action, so a safe-but-useless
solution would pass).

**Common hard-fail patterns.**
- All rubrics in the verifier test format or structure (file types, section
  presence, heading names); none test whether the substance is correct.
- All rubrics check that the agent *did* something (produced output, included
  a section); none check whether doing it achieved the intended outcome.
- Two or more rubrics test the same underlying property (double-counting) —
  the verifier penalises the same failure twice in the aggregate reward.
- The central analytical question of the task has no rubric graders, while
  many peripheral concerns each have dedicated rubrics.
- A rubric checks an aggregate or summary figure while the instruction requires
  per-item detail (per-product, per-component, line-level) — an aggregate-only
  deliverable passes even though the required item-level breakdown is absent.
- The instruction forbids extra outputs ("do not create any additional files",
  "only produce X and Y") but no rubric lists the output directory to fail an
  unexpected artifact, so a side-effect a compliant solution must avoid goes
  ungraded.
- A rubric checks only a negative guardrail (what the solution must NOT do — no
  diagnoses, no off-policy advice, no outside guidelines) while nothing grades
  the positive quality of the required action (was the routing / recommendation
  itself correct and well-reasoned). A solution that safely does nothing useful
  would pass.

---

### verifier_missing_critical_dimensions

```yaml
dimension_name: "Verifier Missing Critical Dimensions"
check_kind: llm
inputs: [verifier_spec_json, instruction, solution_files, verifier_assertion_summary]
applies_when:
  builtin: task_has_verifier
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Is there any specific important quality criterion
that the verifier does not check at all? This is the complement to
`verifier_outcome_coverage` — that dim asks how broad coverage is, this dim
asks specifically what's missing.

**How to judge.** Produce a short list (one or two items) of criteria that a
careful human reviewer of the solution would weight heavily and that the
verifier does not test. If you cannot name anything specific missing, the
severity is 0.

Before flagging a criterion as missing, check whether it is graded implicitly
by a **combination** of existing rubrics, not just by one dedicated rubric. A
concern enforced jointly across several rubrics (e.g. quote feasibility caught
by a compliance gate plus a traceability view plus a cost reconciliation) is
not a coverage gap even though no rubric names it directly. Flag a missing
dimension only when no combination of existing rubrics would catch a solution
that gets that criterion wrong.

Do this trace explicitly: for the criterion you believe is missing, name the
rubric that would FAIL a solution that gets it wrong. If you can name one, it is
covered — do not flag it. "Optimality" / "best-choice" / sub-category-specific
approval concerns are normally enforced jointly by existing selection, cost, and
compliance rubrics; treat them as covered unless a demonstrably wrong choice
would pass every rubric.

When `verifier_assertion_summary` is provided, cross-reference its
per-verifier patterns. If important outcomes are nominally covered by
rubrics that all sit in the "always-passes" list, treat that as effectively
missing coverage of those criteria.

**Severity rubric.**
- **0** — No critical gaps that a careful reviewer would flag.
- **1** — One minor criterion uncovered that probably should be graded.
- **2** — One important criterion uncovered — a criterion a domain reviewer
  would weight heavily.
- **3** — Multiple important criteria uncovered.

**Common hard-fail patterns.**
- The central analytical question of the task (the part that distinguishes
  good solutions from mediocre ones) is not graded by any rubric.
- The most analytically demanding part of the task is ungraded while easier
  surface checks dominate.
- A specific requirement named in the instruction ("recommend a course of
  action") has no corresponding rubric.
- The verifier grades many peripheral aspects but misses the core
  deliverable's primary quality criterion.

---

### verifier_discriminative_power

```yaml
dimension_name: "Verifier Discriminative Power"
check_kind: llm
inputs: [verifier_spec_json, solution_files, verifier_assertion_summary, passed_trial_deliverable_digest]
applies_when:
  builtin: task_has_verifier
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Would this verifier distinguish a good solution from
a plausible-but-wrong solution? If most rubrics would pass for any reasonable
attempt at the task, the verifier has low discriminative power and won't be
useful for ranking model performance.

**How to judge.** Use `verifier_assertion_summary` as the primary signal
when it's available: it lists per-verifier pass rates across multiple
models' trials. The summary's "always-passes" set is direct evidence of
weak discrimination — those rubrics pass for whatever those models
produced, regardless of solution quality. The "discriminates" set is
direct evidence that the verifier IS distinguishing model performance.

An always-fails (0% pass) rubric is the exception: it only counts as weak
discrimination if the golden solution in `solution_files` ALSO fails it. If the
golden passes but the evaluated models do not, the rubric is discriminating
correctly on a hard task — do not count it against discriminative power.

When the summary isn't available (no `evals/` directory present), fall
back to reading each rubric in `verifier_spec_json` and asking: would a
"plausible but wrong" answer (something a competent agent who
fundamentally misread the task would produce) pass it? If yes for most
rubrics, the verifier doesn't discriminate.

**Use the passed-trial exemplar as concrete leniency evidence.** When
`passed_trial_deliverable_digest` is provided, it contains the ACTUAL deliverable
text of the highest-scoring non-crash run plus its per-rubric outcomes. Read the
deliverable and ask: is this top-scoring answer genuinely strong, or did a thin /
partially-wrong solution clear the bar? If a deliverable that misses real
substance still passes the core rubrics, the verifier is too lenient — a
discrimination failure, even when the aggregate pass rates look healthy. Mirror
of the over-strict check: there the concern is a right answer marked wrong; here
it is a weak answer marked right. Do not penalise a legitimately excellent
top-scoring answer — a high reward that the deliverable earns is the verifier
working correctly.

**Severity rubric.**
- **0** — Verifier clearly distinguishes good from plausibly-wrong solutions.
- **1** — Mostly discriminative; one or two rubrics would pass any plausible
  attempt.
- **2** — Several rubrics are too lenient; the verifier mostly accepts.
- **3** — Verifier accepts virtually any plausible attempt; no
  discrimination.

**Common hard-fail patterns.**
- Default-pass rubrics — pass condition is "no obvious problems" or
  "doesn't contain X"; passes any solution that doesn't actively fail.
- Rubrics that pass any output containing certain keywords or section
  headings, regardless of substance.
- Rubrics that pass based on length alone ("the memo is at least 500 words")
  without quality criteria.
- Pass criterion is met by skeleton or placeholder solutions — the rubric
  doesn't require real content.
- Rubrics whose pass condition is satisfied by the formatting itself rather
  than by analytical correctness.

---

## Programmatic Checks (no LLM)

### verifier_spec_parseable

```yaml
dimension_name: "Verifier Spec Parseable"
check_kind: programmatic
callback: check_verifier_parseable
inputs: [verifier_spec_json]
applies_when:
  builtin: task_has_verifier
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
  builtin: task_has_verifier
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
  builtin: task_has_verifier
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

### synthetic_marker_scan

```yaml
dimension_name: "Synthetic Pipeline Marker Scan"
check_kind: programmatic
callback: check_synthetic_markers
inputs: [input_files]
gate:
  role: blocking
  reject_at_severity: 2
  on_disagreement: review
```

**What this evaluates.** Scans the text content of all input files for known synthetic
pipeline markers — bracket-style annotation tags (e.g. `[TIE] Synthetic`,
`[SYNTHETIC]`, `[AI-GENERATED]`) that indicate the task creation tooling's metadata
leaked into the source documents presented to the agent.

This is a deterministic check. A single confirmed marker in any input file produces a
severity-2 finding. There is no count threshold — any confirmed marker in a real-looking
professional document is an unambiguous contamination signal.

**Pass:** No known synthetic markers found in any input file.

**Severity 2:** One or more input files contain a recognised synthetic pipeline marker.
`fixability: fixable` — the file must be cleaned up (marker removed or file regenerated),
but the task itself may be salvageable.

**Marker pattern list** (maintained in `v18_callbacks.py → _SYNTHETIC_MARKER_PATTERNS`):

- `[TIE]`, `[TIE: ...]`
- `[SYNTHETIC]`
- `[AI-GENERATED]`, `[AI_GENERATED]`
- `[LLM-GENERATED]`, `[LLM_GENERATED]`
- `[FABRICATED]`
- `[GENERATED]`
- `[OMEGA]`, `[OMEGA-REBORN]`
- Any `[TAG: value]` pattern matching the above prefixes

To extend the pattern list as new markers are discovered, add to
`_SYNTHETIC_MARKER_PATTERNS` in `v18_callbacks.py`. No spec file changes needed.

**2026-07-27 update:** added two prose-pattern regexes (no bracket tag required)
for a document stating outright it is a fabricated/synthetic/AI-generated
artifact made for training/eval/benchmark purposes — a real miss this dimension
had before (a source PDF read "This document is a FABRICATED artifact created
solely for a financial-analysis training case study" with no bracket tag, so
none of the patterns above fired). Deliberately narrow: only the unambiguous
self-disclosure phrasing, not any broader "sounds like it's talking to an AI"
pattern (rejected during design as unsafe — real business prose legitimately
uses "model", "training", etc. constantly).

---

### near_miss_check

```yaml
dimension_name: "Near-Miss Numeric Consistency"
check_kind: programmatic
callback: check_near_miss
inputs: [input_files, solution_files]
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** The SAME distinctive label (a spreadsheet row, or a
docx/pdf prose line) carrying two values that are within 1% of each other but
NOT equal, at DIFFERENT file/sheet locations — the "same fact rendered
slightly differently" signature (e.g. a reserve figure cited as 81,489 in a
reasoning document but 81,484 in the model and memo it's reasoning about).
Close-but-unequal at different locations is a real reconciliation gap;
genuinely different figures (different periods, scenarios, line items) differ
by more and are not flagged. Ported 2026-07-27 from a QC-skill's
`consistency_ledger.py`, where it was built and validated against real task
files — see that script's docstring for the full exclusion list (calendar
years, same-number-rounded pairs, identifier labels, intermediate-calc
columns, recurring per-entity metrics).

**Pass:** No same-label near-but-unequal value pairs found across files.

**Severity 1 (advisory):** One or more labels carry near-but-unequal values at
different locations. `fixability: fixable`. This stays advisory rather than
blocking because near-but-unequal is a real signal but not by itself proof of
error — the two values' true relationship (a typo vs. two legitimately close
figures) needs a human/LLM read, same as `verifier_rubric_construct_validity`'s
"advisory candidate" precedent elsewhere in this spec.

---

### sum_mismatch_check

```yaml
dimension_name: "Sum-of-Parts Consistency"
check_kind: programmatic
callback: check_sum_mismatch
inputs: [input_files, solution_files]
gate:
  role: blocking
  reject_at_severity: 2
  on_disagreement: review
```

**What this evaluates.** A `total|subtotal|sum` row whose value does not equal
the column-by-column sum of the addend rows above it, computed on the real
spreadsheet grid (reconstructed from the extracted `=== Sheet: X ===` +
tab-separated text). Only fires on a block (rows between blank/header
boundaries) with exactly one total row, as its last row, with 2+ addends —
nested-subtotal blocks are skipped for precision. Ported 2026-07-27 from the
same `consistency_ledger.py`.

**Pass:** No total/subtotal row disagrees with the sum of its own column's
addend rows.

**Severity 2:** A stated total does not equal the sum of its parts by more
than the tolerance (0.5% relative, floor $1). `fixability: fixable`. This is
a pure arithmetic check with a validated low false-positive rate (nested/
duplicated blocks and average/ratio "total" rows are excluded by design), so
it gates as blocking rather than advisory.

---

### ratio_invariant_check

```yaml
dimension_name: "Per-Unit Ratio Invariant"
check_kind: programmatic
callback: check_ratio_invariant
inputs: [input_files, solution_files]
gate:
  role: blocking
  reject_at_severity: 2
  on_disagreement: review
```

**What this evaluates.** A column whose header matches the pattern `<X>_per_
<Y>` (e.g. `Current_GM_per_Facing`) is nonzero on a row where its own `<Y>`
denominator column (e.g. `Current_Facings`) is zero. A per-unit metric is
undefined — not nonzero — over a zero-unit base; this is a pure arithmetic
invariant on the column's own header-declared definition, no domain knowledge
required. Column pairing is header-driven, not a maintained list. Added
2026-07-27 after a confirmed miss: a source workbook had 10 rows with 0
current facings but a nonzero per-facing margin, and the same defect had
propagated into a golden solution workbook at far larger scale (356 rows).

**Pass:** No `<X>_per_<Y>` column is nonzero where its paired `<Y>` denominator
is zero.

**Severity 2:** One or more rows violate the invariant. `fixability: fixable`.
No real false-positive path — the check only compares a column against the
denominator its own header names.

---

### doc_citation_check

```yaml
dimension_name: "Document Citation Consistency"
check_kind: programmatic
callback: check_doc_citation
inputs: [input_files, solution_files]
gate:
  role: blocking
  reject_at_severity: 2
  on_disagreement: review
```

**What this evaluates.** A "Doc NN" / "doc #N" citation (in a workbook's
source-map column, a reasoning file's citation, etc.) whose claimed number
doesn't match the doc number of the file whose title it's actually
describing. Ground truth is real on-disk filenames with a leading number
(`02_Target_Financial_Summary.pdf`) — a citation is flagged only when the
cited label's distinctive words score a clear, unambiguous best match against
a DIFFERENT doc number than the one claimed. A file citing 2+ distinct doc
numbers in one line (e.g. "Doc 01 — Q2 2026 package; Doc 04 debt schedule") is
treated as declaring its own internal doc-numbering legend and exempted for
those specific numbers — a real case found during validation where a model
workbook redefined "Doc 04" for its own purposes; without this guard, 4 of 5
citations flagged in that file were noise downstream of one legitimate (if
sloppy) declared alias. Added 2026-07-27 after confirmed misses: two separate
tasks had a doc-number citation swapped between a workbook's source map and
its own reasoning document.

**Pass:** No citation's title clearly matches a different doc number than the
one claimed.

**Severity 2:** One or more citation mismatches found. `fixability: fixable`.
**Known recall gap, not a precision risk:** short/generic filenames (e.g.
`05_PM_Note.pdf`, whose only content words are filtered as too-generic) have
no distinctive title word left to match against and are silently skipped —
this under-fires rather than false-fires.

---

### deliverable_manifest_check

```yaml
dimension_name: "Deliverable Manifest Match"
check_kind: programmatic
callback: check_deliverable_manifest
inputs: [instruction, solution_files]
gate:
  role: blocking
  reject_at_severity: 2
  on_disagreement: review
```

**What this evaluates.** Extracts literal required deliverable filenames named
in the instruction text (e.g. "produce `forecast.xlsx`" or a
`/workspace/output/report.docx` path) and diffs them against the actual
`solution_files` listing. Flags a required filename that is entirely absent,
present under a different extension or case, or packaged inside an archive
instead of standalone. Pure file-listing comparison — no LLM judgment needed.
Added 2026-07-27 as the single highest-leverage new check from a 12-task
recall audit: nearly a third of the confirmed false negatives found in that
audit were exactly this pattern (a wrong ZIP-vs-individual-files packaging, a
lowercase `.pdf` where `.PDF` was specified, a missing named file, an extra
file beyond what was requested) — all mechanically detectable from a
directory listing alone, which is why this gates as blocking at the same
severity as the other deterministic checks above.

**Pass:** every deliverable filename named in the instruction is present in
`solution_files` under the exact name and extension specified, standalone
(not archived).

**Severity 2:** one or more named deliverables missing, wrong-case, or
archived. `fixability: fixable`. Note: `Reasoning.docx` is excluded from
`solution_files` by this spec's own field binding (surfaced separately as
`solution_reasoning`), so its presence never counts as an unexpected extra
file under this check.

---

### cross_artifact_reconciliation_check

```yaml
dimension_name: "Cross-Artifact Reconciliation"
check_kind: programmatic
callback: check_cross_artifact_reconciliation
inputs: [input_files, solution_files, instruction]
gate:
  role: blocking
  reject_at_severity: 2
  on_disagreement: review
```

**What this evaluates.** Cross-document factual disagreement (a deck's table
disagreeing with a workbook's, a memo's aggregate claim not matching the
source data behind it) and a source document substantially restating the
task's own instructions. This is a `programmatic` dimension by
classification, but internally makes its own LLM calls rather than running a
deterministic check — see the implementation note below for why.

**Why this isn't just a better prompt on an existing dimension.** A single
LLM call asked to "extract claims, then judge consistency" reliably runs the
instruction *nominally* without it changing the verdict — this exact failure
mode was diagnosed and fixed once already, in the QC-skills' equivalent
check. The fix is not more instruction on `solution_internal_consistency`,
`instruction_input_consistency`, or `input_prompt_leakage` — it's **splitting
extraction from judgment into two separate calls, where the second call never
sees the raw documents**, so it can't be swayed by how plausible everything
looks once it's back in context. Call A enumerates every cross-file claim and
any instruction-restating document, blind to judgment. Call B receives ONLY
Call A's claims table — not the source documents — and tries to find a
disagreement for every row before it will call anything "consistent."

Added 2026-07-27 to close the specific gap this pipeline had no other way to
close: 5 confirmed misses (a phantom-defect deck vs. its own root-cause
workbook, two files reporting the same operating metric differently, a
prompt claiming data was fully provided when it wasn't, and a source
document paraphrasing the task's own analysis steps) all needed semantic
cross-document comparison that no deterministic check above can safely do.

**Pass:** Call B finds no genuine same-fact disagreement across any claim,
and Call A finds no document substantially restating the instruction.

**Severity 2:** one or more confirmed disagreements or instruction-restating
documents. `fixability: fixable`. `judges_agreed` is not meaningful here
(single-model, two-call decomposition, not a multi-judge ensemble) — treat
a positive finding as high-confidence given the blind second call, and an
extractor-call failure as a soft "no issue" (fails open, matching the
programmatic-check convention elsewhere in this spec) rather than blocking
the whole sample on an API error.

**Implementation note.** This callback is registered with a 3-parameter
signature (`sample, dim, judge_client_factory`) — `v18_engine.py`'s
`_run_programmatic` detects the extra parameter via `inspect.signature` and
passes the engine's judge client factory to callbacks that declare it,
falling back to the standard 2-argument call for every other programmatic
dimension. This was the minimal change needed to let a `programmatic` dim
make its own LLM calls without introducing a cross-dimension severity-override
mechanism, which the engine does not otherwise have.

---

## Trajectory Quality (LLM, trajectory-reliant)

### intermodel_trajectory_consistency

```yaml
dimension_name: "Intermodel Trajectory Consistency"
check_kind: llm
inputs: [intermodel_trajectory_digest, instruction]
applies_when:
  builtin: task_has_full_trajectories
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Reading the actual model *trajectories* (not just their
scores), does the verifier's reward ordering track the visible quality of the
work, and do the per-rubric outcomes reflect genuine capability differences
rather than a defect in the task or verifier? This is the investigator-agent
lens: several models of known relative strength attempted the same task, and we
inspect HOW each worked to separate "this task is genuinely hard" (real signal)
from "the reward is measuring the wrong thing" (noise).

The `intermodel_trajectory_digest` input contains, side by side for every model:
a mean-reward ranking across its trials, a condensed representative trajectory
(tool histogram, action sequence, deliverables, final message — tool-output bulk
removed), and a per-rubric pass-rate matrix (passes/trials per model) with a
representative judge justification per rubric.

**How to judge.** Look for two distinct failure signatures:

1. **Capability inversion** — a model you have strong reason to consider *more*
   capable produces a clearly more thorough, on-task trajectory (more of the
   required analysis, correct deliverables, sound reasoning) yet scores
   *materially below* a weaker model. Cross-family comparison is fair game here
   because you are reading the trajectories, not trusting raw cross-family
   reward deltas. When the trajectory quality contradicts the reward ordering,
   the reward is likely construct-invalid (rewarding format/luck over substance).
   Within-family inversions (e.g. gpt-5.5 below gpt-5.4, opus below sonnet, a
   newer model below its predecessor) are the strongest form.
2. **Shared identical failure on a core rubric** — several independently capable
   models all fail the *same* `core` rubric in the *same* way (read the
   representative justifications). Two innocent explanations first: the task is
   legitimately hard (the golden still passes — do not flag), or the models all
   made the same real mistake. Flag only when the justifications show the models
   produced defensible, on-task answers that the rubric rejects for an
   unstated/over-strict requirement (over-strict verifier) or for a requirement
   the instruction never conveyed (underspecified prompt). A rubric failing for
   a genuinely missing required output (e.g. "required file missing") is a real
   model failure, not a defect.

**Do not** re-flag a low pass rate as a defect when the spread across models is
wide (some models pass it often) — that is discrimination working as intended.
The signal here is *convergent* failure of capable models against *divergent*
expectations from their trajectories.

**Boundary vs `eval_capability_inversion`.** That deterministic gate checks
within-family mean-reward inversion from the digest alone. This dimension owns
the *cross-family* and *trajectory-grounded* judgment the deterministic gate
deliberately refuses. Do not restate a within-family inversion the deterministic
gate already flags unless the trajectory adds materially to the picture.

**Severity rubric.**
- **0** — Reward ordering is consistent with trajectory quality; per-rubric
  failures reflect real capability gaps or genuinely hard requirements.
- **1** — A minor tension: one rubric or one model pairing looks slightly off,
  but nothing that would change how the task is used as an RL signal.
- **2** — A clear capability inversion OR a core rubric that multiple capable
  models fail for an over-strict/underspecified reason — the reward signal is
  materially distorted and a human should look before this task is used.
- **3** — Pervasive: the reward ordering broadly contradicts trajectory quality,
  or several core rubrics are convergently mis-graded — the task is not a usable
  signal as graded.

**Common hard-fail patterns.**
- A stronger model's trajectory does the required work well but it scores below a
  weaker model whose trajectory is visibly thinner.
- Every model that produced a correct-looking deliverable still fails a core
  rubric whose justification demands something the instruction never stated.
- A core rubric fails for all capable models with justifications that concede the
  answer is substantively right but penalise wording/format/an extra detail.
- The customtools / harness variant of a model fails file-production rubrics that
  its siblings pass ("required file missing"), suggesting a harness artifact
  rather than a capability difference — surface for human triage.

**Anti-overcall.** A wide pass-rate spread is healthy discrimination, not a
defect. A uniformly hard rubric that the *golden* passes is legitimate
difficulty. Genuine "required output absent" failures are real. Reserve
severity 2+ for cases where the trajectories actively contradict the reward.

**Boundary vs `prompt_underspecification` and `prompt_misleading`.** Those two
dims locate the fault in the PROMPT (a legitimate rubric the instruction failed
to convey, or actively steered away from). This dim locates it in the
VERIFIER/reward (a rubric that is itself wrong, or a reward that contradicts
trajectory quality). If the rubric requirement is reasonable and the fix is to
edit the instruction, route it to those dims, not here.

---

### prompt_underspecification

```yaml
dimension_name: "Prompt Underspecification"
check_kind: llm
inputs: [instruction, input_files, verifier_spec_json, intermodel_trajectory_digest]
applies_when:
  builtin: task_has_full_trajectories
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Does the verifier enforce a *legitimate* requirement
that the instruction never actually told the model to satisfy? This is the
"silent omission" failure: the requirement is fair to grade, but a capable model
had no reasonable way to infer it from the prompt + inputs — so it is unfairly
penalised for a gap the task author left, not a capability gap.

**How to judge.** Use `intermodel_trajectory_digest`: find a core rubric that
*multiple capable models fail convergently* (read the per-rubric matrix and the
representative justifications). Then go back to `instruction` (and `input_files`)
and ask: **could a competent professional have inferred this requirement from
what the prompt actually says?** If yes → not underspecified (the models simply
missed it; that is real signal). If no → the prompt is underspecified: the
requirement lives only in the hidden rubric. Confirm the requirement is otherwise
reasonable (if it is *not* reasonable, that is over-strictness →
`intermodel_trajectory_consistency`, not here).

The distinguishing test: **the fix is to add a sentence to the instruction**, and
with that sentence a capable model would have complied. That is underspecification.

**Evidence bar for severity 2+ (to keep this judgment stable).** Do not reach
severity 2 unless ALL of the following hold, each with concrete evidence you can
cite:
1. the failing rubric is `core` (importance core), not secondary;
2. it is failed by **at least two capable models** in the digest (read the matrix,
   not one trial);
3. you can **name the specific element** the rubric demands (a figure, statement,
   tab, count); and
4. you have **checked the instruction and inputs and that element is genuinely
   absent** from what a competent reader is told to produce.
If any of the four is uncertain, cap at severity 1. **When torn between 0 and 2,
choose the lower.** Reserve severity 2+ for evidence strong enough that you'd
state it with high confidence; a hedged "might be underspecified" is severity 1.

**Severity rubric.**
- **0** — Every graded requirement is inferable from the prompt and inputs (or
  the four-point bar above is not clearly met).
- **1** — A minor/secondary requirement is under-stated, OR a core requirement is
  only *possibly* underspecified (evidence is suggestive, not conclusive) —
  unlikely to change how models are ranked.
- **2** — All four conditions above hold for one core requirement: it is not
  inferable from the prompt and capable models convergently fail it for that
  reason — the reward penalises a prompt gap, not capability.
- **3** — Multiple core requirements independently meet the four-point bar; the
  task can only be solved by guessing the hidden rubric.

**Common hard-fail patterns.**
- A core rubric all capable models fail demands an output element (a specific
  figure, statement, tab, count) the instruction never asks for.
- The justifications repeatedly cite a missing item that appears nowhere in the
  prompt or inputs.
- The golden contains the element only because the author knew the rubric — a
  fresh reader of the instruction would not produce it.

**Anti-overcall.** Requirements that are standard professional practice for the
stated deliverable ARE inferable — do not flag them. Eval-context conventions
(named files, output paths) are not underspecification. If even one capable model
inferred and satisfied the requirement, it was inferable.

---

### prompt_misleading

```yaml
dimension_name: "Prompt Misleading"
check_kind: llm
inputs: [instruction, input_files, verifier_spec_json, intermodel_trajectory_digest]
applies_when:
  builtin: task_has_full_trajectories
gate:
  role: advisory
  reject_at_severity: 3
  on_disagreement: review
```

**What this evaluates.** Does the instruction actively steer models toward an
approach, framing, or answer that the verifier then penalises? This is the
"active misdirection" failure — stronger than mere omission: the prompt does not
just leave a gap, it points the wrong way, so following it faithfully lowers the
reward.

**How to judge.** Look in `intermodel_trajectory_digest` for capable models whose
trajectories show them doing what the prompt appears to ask, yet failing a core
rubric for it. Then read `instruction`: is there wording that reasonably leads to
the penalised approach (a suggested method, an example that conflicts with the
rubric, an emphasis that mis-prioritises, a definition at odds with the grader)?
If following the instruction as written *causes* the rubric failure, the prompt
is misleading.

**Boundary vs `prompt_underspecification`.** Underspecification is SILENCE (the
prompt says nothing about the requirement). Misleading is MISDIRECTION (the prompt
says something that points away from it). If the prompt is merely quiet, use the
other dim; only flag here when specific wording does the steering. Do not flag
both for the same root cause.

**Evidence bar for severity 2+ (to keep this judgment stable).** Misdirection is
a strong claim; require strong evidence:
1. you can **quote the exact instruction phrase** that steers toward the penalised
   approach (a suggested method, a conflicting example/definition, a
   mis-prioritising emphasis) — a specific span, not a paraphrase;
2. you can **name the core rubric** it conflicts with; and
3. the digest shows **capable models actually followed that wording and failed
   that rubric for it** — the steering demonstrably caused the failure.
**If you cannot quote a specific steering phrase, this is not misleading —
severity 0** (if a requirement is merely absent, that is
`prompt_underspecification`, not this dim). **Silence is never misleading.** When
torn between 0 and 2, choose the lower; a hedged claim is at most severity 1.

**Severity rubric.**
- **0** — Nothing in the instruction steers models away from a high reward (or no
  specific steering phrase can be quoted — the default).
- **1** — Slightly ambiguous phrasing that a careful reader would still get right,
  OR steering you suspect but cannot pin to a quoted span.
- **2** — All three conditions above hold: a quotable instruction phrase leads
  capable models to an approach a core rubric penalises — the prompt is fighting
  its own verifier.
- **3** — The instruction's central framing contradicts what the verifier
  rewards; faithful compliance reliably fails.

**Common hard-fail patterns.**
- The prompt suggests a method/format the rubric marks wrong.
- An example or definition in the instruction conflicts with the grader's
  expectation.
- Emphasis in the prompt mis-prioritises, and models spend effort where the
  rubric gives no credit while missing what it does.

**Anti-overcall.** A prompt that is merely hard, or that withholds a requirement
without pointing the wrong way, is not misleading (that is underspecification or
legitimate difficulty). Reserve this for wording that demonstrably causes the
failure the trajectories show.

---

## Synthesis (framework-managed, not a dimension)

After all per-sample dimensions are graded, the synthesizer makes a single
LLM call to produce the `SampleReview` artifact. The synthesizer is given:

- The full ordered list of dimension findings with severity ≥ 1 (pre-sorted by
  `(gate.role, -severity, -fixability_weight)`)
- The spec's gating rules (so it knows which dims are blocking)
- The verdict-tier definitions

It returns:

```
SampleReview(
    verdict          : ACCEPT | ACCEPT_WITH_FIXES | NEEDS_REVIEW | REJECT_FIXABLE | REJECT_STRUCTURAL
    one_line         : str
    top_issues       : list[Issue]                # capped at synthesis.top_issues_cap
    suggested_fixes  : list[ConcreteFix]
    review_required_because : str | None
)
```

The synthesizer is permitted to invent issue categories within reason, but
must not invent severity levels that don't appear in the underlying findings.
It may *promote* an issue above the default ranking if it provides a one-line
rationale; it may *not* hide a blocking finding.

## Batch summary

The framework emits `batch_summary.md` containing:

- `verdict_distribution` — count and percentage in each of the five verdict
  tiers
- `failure_mode_histogram` — frequency of each issue category appearing in
  `top_issues` across the batch
- `judge_agreement_rate` — fraction of (sample × dim) pairs where the two
  judges' severities differed by ≤ 1
- `calibration_drift` — for any calibration golden whose verdict changed
  relative to the recorded expected verdict

## Output artifacts

- `batch_review.json` — structured backend, source of truth (one entry per
  sample with verdict, top_issues, all_findings, judge_trace_path)
- `reviewer_queue.csv` — flat spreadsheet view with fixed-width issue slot
  columns; sorted with NEEDS_REVIEW samples at the top
- `batch_summary.md` — the four batch-level metrics above
- `raw/sample_<idx>.json` — full per-judge transcripts, for audit drill-down
