# Second development batch: routing imports and deadline headroom

## Decision

Retain `--lazy-router-imports` as an optional engineering improvement for independent
evaluation. Keep every production default unchanged. In 20 paired comparisons the
median external saving was **1.758 seconds**, meeting the predeclared one-second
target. All 20 pairs had identical prediction hashes and model-selection evidence;
all six generous-budget pairs also had identical complete routing plans. This is
useful deadline headroom, but **did not increase strong-model completion in this
study**. The wider 25-second fixture still recovered its weak baseline once in
five runs under both routed variants. Auto completed the stronger model in all
five contemporaneous runs.

No official grading metric or maximum-size certification is available. Accuracy,
balanced accuracy, macro F1 and minority recall below are provisional measures.

## Changes and causal isolation

The experimental CLI flag loads the same strict router configuration through a
new standard-library-only module. The normal CLI continues importing `routing`
in the parent. The public `routing.load_router_config` remains compatible and
delegates to the same loader. No estimator, score, candidate order, weighting,
preprocessing, checkpoint, worker isolation or regression model was changed by
this optimization. The opt-in `--timings` instrumentation now separates parent
data imports, router imports and configuration parsing.

The median recorded parent router-import phase fell from **1.828 seconds to
0.000 seconds** (timer resolution rounds very small values to zero). This
isolates the avoided modeling-stack import from configuration parsing, which
remained negligible. The overall paired saving of 1.758 seconds is close to
that reduction. Startup, scheduling, worker imports, fitting and output remain
separate costs; this result does not attribute the entire historical residual
to imports. The ordinary parent still needs pandas/NumPy for contract/profile
work. Worker imports are retained.

## Frozen development protocol and reproducibility

- Identical 96x8 and 660x110 training fixtures from the first-batch generator,
  each with 120 evaluation rows and 10% random feature missingness.
- Current routed, lazy routed and auto; internal budgets 10/25/50 seconds.
  Three repetitions per cell, with wider/25 extended to five: exactly 60 runs.
- Serial randomized policy order, seed 10490; native numerical threads limited
  to one; reused offline monitor with an external 60-second guard.
- All production/configuration sources were frozen before timing and checked
  before every run. `runtime-source.zip` preserves their exact bytes plus both
  harness sources (archive SHA256
  `b006f03d42078f8806def4fb665d50ef570f3d93d9f9f8caa66ff42d574c5bab`).
- Segment one saved 40 runs in 301.875 seconds. The exact unrun suffix saved 20
  in 399.922 seconds; no completed key was rerun. Input/source/config hashes and
  references match across segments. All 60 keys are unique.
- Evaluation labels live in the offline harness and are absent from CLI inputs.
  Full reference/prediction arrays are saved; all four metrics were independently
  recomputed from those arrays using standard-library arithmetic.
- Python/package environment: NumPy 2.3.5, pandas 3.0.1, SciPy 1.18.1,
  scikit-learn 1.9.1. Python/platform/CPU information is in the raw segments.
  Process-tree RSS remains unavailable in this environment; no memory-limit
  certification is claimed.

Evidence: `experiments/second-batch/runtime-study.json`,
`runtime-supplement.json`, `runtime-summary.json` and `runtime-source.zip`.
The summary records every paired timing difference and selection comparison.
To reproduce on a fresh output path:

```powershell
.\.venv312\Scripts\python.exe scripts/second_runtime.py --output experiments/second-batch/runtime-new.json --max-seconds 900
```

Use `--resume-from` with a new report path for an explicitly unrun suffix; source
or input mismatches are rejected. Historical report paths cannot be overwritten.

## Quality, time, validity and fallback together

All **60/60 outputs were valid**, with no external timeout and maximum external
wall time **22.531 seconds**. Every routed/lazy paired prediction hash, selected
candidate, validation score, objective, weight, candidate error and skip field
matched. All six 50-second routing plans matched too. Accuracy, balanced accuracy,
macro F1 and minority recall therefore show no routed/lazy regression on these
fixtures, including the tight budgets. Repeated timings varied: paired savings
ranged from 0.077 to 3.281 seconds, so a one-second saving is not guaranteed for
every execution.

Table entries are median external seconds, in current routed / lazy routed / auto
order. BA entries are repeated-run means in the same order. Recovery counts are
natural completed-baseline recoveries, not deliberately injected failures.

| Fixture | Internal budget | Median external seconds | Mean BA | Recoveries |
|---|---:|---|---|---|
| Normal | 10 | 7.360 / 5.922 / 5.281 | .77429 / .77429 / .77429 | 0 / 0 / 0 |
| Normal | 25 | 7.188 / 5.188 / 4.938 | .87983 / .87983 / .87983 | 0 / 0 / 0 |
| Normal | 50 | 7.297 / 5.765 / 5.703 | .87983 / .87983 / .87983 | 0 / 0 / 0 |
| Wider | 10 | 8.329 / 5.406 / 5.234 | .50000 / .50000 / .50000 | 0 / 0 / 0 |
| Wider | 25 | 21.281 / 19.531 / 18.390 | .80739 / .80739 / .88424 | 1 / 1 / 0 |
| Wider | 50 | 20.437 / 19.563 / 18.468 | .88424 / .88424 / .88424 | 0 / 0 / 0 |

The median of paired time savings (distinct from subtracting independent group
medians) is 1.438/1.922/1.673 seconds on normal at 10/25/50 seconds, and
1.922/1.313/1.843 seconds on wider at 10/25/50 seconds.

Normal 10-second runs choose Extra Trees. Normal 25/50-second runs choose logistic
regression. Wider 10-second runs choose Extra Trees and have zero minority recall.
Wider 50-second runs all choose histogram boosting. At wider/25, current and lazy
each finish histogram boosting in four of five repeats and recover Extra Trees
in the same remaining repeat. Their mean accuracy is .85167, macro F1 .78234 and
minority recall .69189, compared with auto's .89167/.87571/.86486. Histogram
finishing is therefore still materially useful. A valid baseline recovery is not
a claim of good predictive quality.

Unlike the first batch, the current routed strategy completes histogram boosting
in most wider/25 runs. Comparing historical timing means would confound host
conditions. Only the randomized contemporaneous pairs support the import-change
claim. This study confirms recovery remains available, rather than measuring a
positive predictive-quality benefit from recovery on these two fixtures.

## Correctness checks and remaining work

**56 focused tests passed** in 66.64 seconds: new import-isolation/configuration/CLI
checks and the existing timing, checkpoint-recovery and routing tests. The isolated
configuration subprocess asserts sklearn, NumPy and pandas are not imported. CLI
use with baseline/auto is rejected. Explicit missing configuration files and
unknown keys still fail. All default prediction/selection compatibility and
regression integration checks are additionally coordinated by the main agent;
the 60-run runtime experiment itself contains classification fixtures only.

The one-second speed criterion and generous-budget invariance criterion pass.
The stronger-model completion benefit remains **unconfirmed**. Retain checkpoints
and isolation. A separately controlled, optional classification-only refit-cost
admission experiment is justified: when the best completed validation candidate
cannot plausibly refit before the hard deadline, consider the next best completed
candidate instead of losing all optional work. Admission must use training-only
scores and measured costs, preserve regression/defaults and retain histogram's
50-second performance. The worker's advertised budget currently precedes its
spawn/import overhead; an admission estimate must account for the actual parent
deadline rather than assume that soft budget is wholly available to fitting.

No default promotion, confirmation campaign, commit or push was performed by this
runtime investigation. Two related fixtures do not establish unseen-family
generalization, maximum-size runtime compliance or official contest points.
