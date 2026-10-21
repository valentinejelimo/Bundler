---
name: gdpval_svc_noeval_qc
extends: svc.md
# NO-EVAL variant of the SVC front: same layout remap and dim overrides as
# svc.md, minus reliance on eval-trial signal. The 4 verifier-quality LLM
# dims (construct_validity, outcome_coverage, missing_critical_dimensions,
# discriminative_power) read verifier_assertion_summary / the trial-deliverable
# digests — without evals/<model>/trial_*/ present those fields are empty and
# the judgment would be ungrounded, so they're re-gated to task_has_eval_runs
# (False when no eval data) instead of task_has_verifier (True whenever
# tests/rubric.json is non-empty, which it always is). Everything else —
# path remap, task_is_substantive gates, the 6th LLM dim (verifier_rubric_clarity),
# the 3 programmatic-check skips — is inherited unchanged from svc.md.
---

### verifier_rubric_construct_validity

```yaml
dimension_name: "Verifier Rubric Construct Validity"
check_kind: llm
inputs: [verifier_spec_json, instruction, solution_files, verifier_assertion_summary, failed_trial_deliverable_digest]
applies_when:
  builtin: task_has_eval_runs   # NO-EVAL variant: gate off when evals absent
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
  builtin: task_has_eval_runs   # NO-EVAL variant: gate off when evals absent
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
  builtin: task_has_eval_runs   # NO-EVAL variant: gate off when evals absent
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
  builtin: task_has_eval_runs   # NO-EVAL variant: gate off when evals absent
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
