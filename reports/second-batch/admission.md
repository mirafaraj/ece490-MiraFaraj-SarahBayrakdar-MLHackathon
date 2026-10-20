# Separate development experiment: refit-cost admission

## Decision

**Reject this fixed admission policy as a quality improvement.** At a 25-second
internal budget it saved time by refusing histogram refits that the contemporaneous
unguarded strategy completed successfully. Every classification measure worsened
in all five pairs. The 50-second histogram gains were preserved, but that does not
justify the tight-budget losses. Keep this control off; its measured failure is
useful development evidence, not a promoted model. Do not retune its factor on the
same inspected references and present a rerun as confirmation.

## What changed and why

The first 60-call import comparison still had one weak baseline recovery in five
wider/25 runs under both routed variants. This motivated a **separate** 16-call
experiment. The optional `experimental.refit_admission` control keeps the existing
three estimator families and training-only validation scores, then filters refits
using measured validation-fit cost and actual parent-deadline headroom. It uses
the best feasible completed candidate, reusing the baseline if necessary.

The fixed cost multiplier **1.5** was declared from the earlier first-batch fit
ratios before these calls, as recorded in `admission-protocol.md`. The available
headroom is the absolute parent worker deadline minus current time and the existing
five-second plan reserve. That absolute deadline includes spawning/imports and the
parent's output reserve. Conditional propagation changes the opted-in routed
classification path only; the option defaults to false, and regression and ordinary
selection paths remain unchanged. The main agent owns implementation/correctness
tests; this report covers the frozen CLI experiment.

## Frozen protocol and evidence

Both variants use lazy imports. Compare explicit router overrides
`{"experimental":{"refit_admission":false}}` and `true` on the same wider
660x110/120-row development fixture: five paired repeats at 25 seconds and three at
50 seconds. Randomized serial policy order uses seed 20490. Native threads are one;
the external guard is 60 seconds. All 16 calls completed in one frozen segment in
292.422 seconds; there were no reruns or factor adjustments.

`scripts/second_admission.py` saved source/configuration/input hashes, raw
references/predictions, metrics, profiles, model-selection/fallback/eligibility
metadata and external times. Source/configuration hashes were rechecked before
every call. Exact source bytes are archived in
`experiments/second-batch/runtime-admission-source.zip` (SHA256
`7d0b7dc58ce0d58114b8f378d3af325eb5acb0da7d2a40730965f96b38029754`).
Raw evidence is `runtime-admission-study.json`; checked summaries/pairs are in
`runtime-admission-summary.json`. All four metrics were reconstructed from saved
arrays with standard-library arithmetic. References remain offline; they do not
enter the input directory, admission estimate or model selection.

These 16 runs use a different frozen implementation from the original 60 calls.
Their timings and fallback frequencies must be compared contemporaneously and
reported separately. In particular, the unguarded strategy had no recovery in
these five 25-second runs; the earlier 1/5 frequency is not a guaranteed failure
rate. RSS is unavailable. This is one development fixture, not maximum-size or
assessment-environment certification.

## Before versus after

The table compares lazy routed without / with admission. Times are median external
seconds; quality values are repeated-run means. Metrics remain provisional; no
official accuracy/BA preference is assumed.

| Internal budget | External seconds | Accuracy | Balanced accuracy | Macro F1 | Minority recall | Selected models |
|---|---|---|---|---|---|---|
| 25 | 19.859 / 13.125 | .89167 / .79167 | .88424 / .75204 | .87571 / .75392 | .86486 / .64865 | Histogram 5/5 / linear 5/5 |
| 50 | 21.782 / 21.485 | .89167 / .89167 | .88424 / .88424 | .87571 / .87571 | .86486 / .86486 | Histogram 3/3 / histogram 3/3 |

All **16/16 outputs were valid**; there were zero implementation failures, optional
stage errors, external timeouts, fallbacks or checkpoint recoveries. Maximum
external time was **21.891 seconds**. The harness separately flags `optional_error`
as an implementation failure even if valid baseline output exists. Each of the
eight admitted completed selections records `deadline_source=parent_absolute`
and full eligible/rejected/before/after diagnostics.

All eight paired validation-score dictionaries were identical. At 25 seconds the
guard always rejected histogram and selected linear; the unguarded variant
always completed histogram. Predictions and the resulting quality loss therefore
reproduced in all five pairs: accuracy **-10.00 points**, BA **-13.22 points**,
macro F1 **-12.18 points**, minority recall **-21.62 points**. The median paired
time saving was 6.905 seconds, which cannot justify those quality losses.
At 50 seconds all three pairs preserve the histogram candidate, validation scores
and prediction hashes. Paired timing differences range from -1.610 to +1.985
seconds there; do not claim an additional 50-second speed improvement from this
small, noisy comparison.

## Explanation and next step

The guard functions correctly but is too conservative for this fixture under
current host conditions. For example, it estimates roughly 9.87 seconds for a
histogram refit against about 4.89 seconds of reserved headroom and switches to
linear, while the paired unguarded histogram finishes before the parent deadline.
The actual deadline propagation fixes optimistic bookkeeping; the fixed multiplier
plus reserve does not yet produce a beneficial prediction policy.

A later, separately approved experiment should investigate improving the **quality
of completed recovery checkpoints** while retaining useful histogram attempts.
One hypothesis is to fit/publish an inexpensive full-data linear backup only when
its training-only score beats the completed baseline and its measured cost fits,
then try histogram. Such a change may preserve successful histogram predictions
and improve failed-refit recovery, but adds work and can itself increase misses.
It requires its own cost/validity/quality measurements; it was not implemented in
this batch. No further experiment, default promotion, commit or push followed
these results.
