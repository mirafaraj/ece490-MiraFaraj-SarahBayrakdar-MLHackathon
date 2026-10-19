# Evaluation policy: retain useful improvements and expose tradeoffs

## Current user preference (supersedes the strict policy below)

The latest instruction is to retain steps that improve overall evaluation and
performance without material negative impact. Ordinary accuracy remains important,
but a drop in that metric alone must not discard improvements in other tasks or
an otherwise useful candidate. The earlier strict policy below is retained as a
historical/optional no-drop diagnostic, not the default decision rule.

- Evaluate classification, regression and clustering separately. Retain useful
  task policies even when another task needs revision.
- Identify clear improvements, equivalent results, tradeoffs, dominated choices,
  and insufficient evidence. Report both ordinary and balanced accuracy, worst
  case/family losses, time, validity, memory evidence and recovery behavior.
- Metric decreases are negative effects to quantify, not something to rename
  "no impact." A justified tradeoff can still be useful, but requires an explicit
  objective and predeclared loss tolerances. Default tolerances are zero until
  chosen on development data; do not backfit them to an observed confirmation loss.
- A candidate with a tradeoff can be retained experimentally while promotion
  awaits official scoring and fresh evidence. An observed metric loss does not
  mean deleting its code or rejecting unrelated regression gains.
- Invalid output, corrupted provenance, label leakage and hard resource failures
  remain blockers. An invented combined score cannot justify them.
- Unknown official normalization means no defensible single overall grade exists
  yet. Compare separate task metrics and quality/runtime/reliability outcomes.

The tradeoff-aware evaluator and latest evidence take precedence over historical
strict verdicts. `--enforce-accuracy-gate` remains available when an explicitly
requested no-drop experiment is appropriate.

## Historical strict policy and integration experiment

User requirement, 19 October 2026: improve the model and grading prospects without
reducing ordinary classification accuracy. This supersedes any earlier proposed
gate that would tolerate classification accuracy losses in exchange for balanced
accuracy gains.

## What we can enforce

Compare each frozen candidate against the same frozen baseline on identical
evaluation inputs. Reject the candidate if ordinary accuracy decreases on any
measured classification case, beyond 1e-12 floating-point comparison tolerance.
An improving overall mean cannot cancel one failing case. Also report per-family
and overall accuracy, balanced accuracy, validity, fallback use, and timing.

Keep the baseline as the default until all required promotion checks pass. The
gate evaluates a deployable model-selection policy, not an oracle that chooses a
model after reading evaluation labels. The evaluated baseline must be identified
by source/config hashes and retained across improvement iterations; moving the
baseline to a weaker candidate would undermine this protection.

This is an empirical constraint on recorded datasets. Neither training validation
nor finite benchmarks guarantee no accuracy loss on an unseen private dataset.
The result must say "no measured accuracy regression" rather than "accuracy can
never drop." Choosing the unchanged baseline preserves its predictions, but a
different learned model has no general hidden-data dominance guarantee.

## Evidence required before promotion

1. Full paired inputs, finite valid metrics, complete reports, stable source and
   configuration provenance, and no unexplained missing cases.
2. No measured ordinary-accuracy loss on any classification case; balanced
   accuracy remains a secondary diagnostic even when it is the official objective.
3. No invalid output or new emergency fallback. Distinguish completed-baseline
   checkpoint recovery from dummy prediction fallback and evaluate their quality.
4. A frozen candidate evaluated once on fresh, previously unopened confirmation
   cases/settings. Previously inspected seeds are development evidence.
5. Alignment with official metrics and profile/contract requirements, plus the
   published runtime/memory limits. Passing the accuracy gate alone does not mean
   a candidate is ready for submission or will maximize grades.

Missing evidence means insufficient evidence, not a pass. A known violation is a
rejection even when some other evidence is missing. All-task promotion also needs
regression, clustering, profiling, and runtime checks; an accuracy-only report
cannot authorize it.

## Integration recommendation for the model owner

Current routed supervised code supports objective-aware class weighting: accuracy
uses unweighted estimators and balanced accuracy uses balanced weights. This is
an implementation capability, not measured proof that the routed strategy meets
the requirement. The historical auto strategy selected for balanced accuracy and
fails the requirement on already evaluated imbalanced cases.

Before changing the routed selector, measure its frozen current behavior. If
necessary, compare every optional classifier and the baseline on the same
training-only validation folds. Retain a candidate only if it meets the ordinary
accuracy constraint relative to that baseline; optimize the configured official
metric among survivors. If no optional candidate qualifies, use the completed
baseline. Log both validation metrics, comparison baseline, folds and rejection
reason. Do not fit the baseline on the validation labels and score it on those
same rows: fit each fold's baseline only on that fold's fitting subset.

Validation folds never include benchmark references or private test labels.
Retain full-training baseline predictions as the runtime checkpoint. A fold-level
constraint reduces observed validation risk; the independent outer promotion gate
must still check the final refitted policy on untouched evaluation rows.

The implementation chat currently owns these production files and is integrating
and timing the router. This chat changes only the independent evaluator and its
documents, avoiding overlapping writes and competing model runs.

`accuracy-objective.json` is a reviewable experimental override for the existing
router's partial-configuration API. It selects ordinary accuracy and retains zero
near-tie tolerance; other settings come from versioned built-in defaults. It does
not change repository defaults or implement a dual-metric validation constraint.
Only run the following experiment after the implementation owner's measurements
finish and a source/configuration freeze is recorded. Use a unique output path:

```powershell
.\.venv312\Scripts\python.exe -u scripts/evaluate.py --suite router --seeds 42 314 --strategies baseline routed --router-config reports/grade-evaluation/accuracy-objective.json --output reports/grade-evaluation/accuracy-objective-development.json
.\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py reports/grade-evaluation/accuracy-objective-development.json --candidate-strategy routed --enforce-accuracy-gate --output reports/grade-evaluation/accuracy-objective-assessment.json
```

These reused seeds are development only. An accuracy objective is a hypothesis
for improvement, not a promise of no loss relative to the unchanged baseline.

## Recommendations must be evidence-based

Each evaluator recommendation should identify a failing case/family and metric,
state a plausible cause as a hypothesis, specify one controlled change, list
controls and acceptance checks, and report what evidence is missing. Prioritize
official model quality (85 points shared equally across three tasks), profiling
(10), then conditional speed credit (5). Do not convert raw accuracy, RMSE and ARI
into an invented total grade. Reject improvements that violate this accuracy
constraint even if a provisional proxy score increases.
