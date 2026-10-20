# Predeclared separate refit-admission protocol

This is a separate optional development experiment, after the frozen 60-call
import comparison. Never pool timings across the two implementations. Production
defaults and regression behavior remain unchanged. No default promotion follows.

The first batch's three completed wider histogram refits had final-fit/validation-
candidate time ratios 1.02896, 1.02713 and 1.14501. Predeclare a factor of **1.5**,
rounding that historical maximum up to the next half unit. This factor is chosen
before the new 16 runs, without new evaluation labels. Actual hard-deadline
headroom must account for parent/worker imports and spawning. Rank candidates by
the existing training-only validation score among those with plausible remaining
refit cost; prefer a completed linear alternative over a weak baseline when
histogram cannot plausibly finish. Retain the existing completed baseline and
checkpoint as recovery options. The main agent owns implementation details/tests.

`scripts/second_admission.py` compares lazy routed with `refit_admission=false`
versus `true`, with all other controls held equal. Use the existing wider fixture
(660x110 training, 120 evaluation rows), five paired repeats at 25 seconds and
three at 50 seconds: **16 serial CLI runs**. Policy order is randomized with seed
20490. Native threads are limited to one; external guard is 60 seconds. Freeze
sources/configurations before the first call and archive exact bytes in
`runtime-admission-source.zip`. Save full references, predictions, metrics,
configuration/source/input hashes, model metadata, fallback/recovery flags and
wall time per run. Labels remain offline and do not enter selection/configuration.
Reject overwritten evidence; bounded segments may resume only their exact unrun
suffix with identical source/config/input hashes. Do not tune the factor on these
results and rerun the same fixture as confirmation.

Retention requires valid output and intact recovery, no external 60-second breach,
and preserved histogram selection/predictions at 50 seconds. Judge 25-second
benefit by actual quality and fallback outcomes across pairs, reporting accuracy,
balanced accuracy, macro F1 and minority recall together; no official metric is
assumed. Reduced wall time alone cannot justify suppressing a stronger model.
Unchanged or worsened quality is inconclusive/unsuccessful for score improvement,
even if the engineering guard is correct. One fixture cannot establish unseen-
dataset superiority; promising behavior requires later independent evaluation.
