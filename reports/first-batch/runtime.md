# First development batch: runtime and checkpoint recovery

## Decision

Keep checkpoint recovery. Do not promote a model or change a production default
from this experiment. The measurements identify parent startup/configuration work
and expensive histogram selection/refitting as better optimization targets than
checkpoint serialization. Routing can materially reduce useful model completion
at a short internal budget; recovering a weak baseline preserves valid output but
does not automatically preserve predictive quality.

## Implemented controls and scope

`main.py --timings` opts into diagnostic metadata. It does not enable experimental
model controls. With the flag absent, timing fields are omitted. The runner records
input loading, profiling, worker spawn/wait, result validation/recovery, and worker
import/fit totals. Final output serialization duration is in the final stderr JSON
status, so reporting it does not require rewriting output artifacts.

Routed diagnostics additionally record plan construction, completed-baseline work,
checkpoint publication, and optional work. Supervised optional work exposes
selection/refit/prediction times; clustering exposes its own phase timings. The
metadata deliberately retains these component-specific names. Fine-grained
baseline/auto preprocessing and selection phases and worker pickle serialization
are not isolated. Recovered checkpoints contain only phases completed when the
checkpoint was published; missing optional timings must not be interpreted as zero.

The offline harness ran the fixed 54-run protocol: baseline/auto/routed, internal
budgets 10/25/50 seconds, three repetitions, and two deterministic classification
fixtures of 96x8 and 660x110 training features. Each has 120 evaluation rows, 10%
random feature missingness, a noisy linear signal, and majority prevalence around
three quarters. These are development fixtures, not independent confirmation
families or evidence about unpublished maximum assessment shapes. Evaluation
labels live only in the harness and never enter the input directory or selector.

Strategy order was randomized within each paired repetition with seed 490. Runs
were serial, with native thread limits of one and an external 60-second guard.
The initial 300-second start cap completed 39 rows in 309.500 seconds; an exact
unrun-suffix supplement completed nine in 155.296 seconds; the final six took
111.672 seconds including the controlled recovery demonstration. No completed
row was rerun. All 54 planned keys are present exactly once. The offline harness
gained resume support during this process; its different hashes are preserved.
Production/configuration hashes match across all three artifacts and the final
working files, and generated fixture bytes match across supplements.

Raw evidence is in `experiments/first-batch/runtime-study.json`,
`runtime-supplement.json`, and `runtime-supplement-2.json`; the checked union is
`runtime-combined.json`. Every report path is protected against overwrite by the
harness. They retain source/config/input hashes, environment, order, metrics,
errors, prediction hashes, model metadata, and timing status.

## Quality, wall time, validity, and fallback together

All 54 outputs were valid. The maximum measured external wall time was 27.078
seconds, with no external 60-second breach. This demonstrates headroom for these
two fixtures only; it does not establish official maximum-size compliance.

The following are three-repeat means. Wall-time triples and BA triples are in
baseline / auto / routed order. BA is provisional, as are the other metrics.

| Fixture | Internal budget | External wall seconds | Balanced accuracy | Natural fallback outcomes |
|---|---:|---|---|---|
| Normal | 10 | 6.328 / 6.151 / 8.724 | .7743 / .7743 / .7743 | Routed: 1 completed-baseline recovery |
| Normal | 25 | 6.005 / 6.969 / 9.672 | .8406 / .8798 / .8798 | None |
| Normal | 50 | 6.005 / 6.000 / 8.755 | .8406 / .8798 / .8798 | None |
| Wider | 10 | 6.359 / 6.182 / 8.786 | .5000 / .5000 / .5000 | Routed: 1 emergency fallback |
| Wider | 25 | 6.078 / 21.969 / 22.531 | .5000 / .7562 / .5000 | Auto: 1 emergency; routed: 3 completed-baseline recoveries |
| Wider | 50 | 6.823 / 21.719 / 26.203 | .5000 / .8842 / .8842 | None |

At normal 25/50 budgets, selecting logistic regression increases BA from .8406 to
.8798 and minority recall from .7273 to .9091, but decreases ordinary accuracy
from .8917 to .8667 and macro F1 from .8571 to .8460. These conflicting metrics
cannot justify an official-score claim.

At wider 50, auto and routed select histogram boosting and agree byte-for-byte
in all three repetitions: accuracy .8917, BA .8842, macro F1 .8757, minority recall
.8649. The baseline predicts the majority class throughout: accuracy .6917,
BA .5000, macro F1 .4089, minority recall zero.

At wider 25, auto completes the same useful histogram model in two repetitions
and falls back to the majority class in one. Its averaged accuracy/BA/F1/minority
recall are .8250/.7562/.7201/.5766. Routed recovers its completed baseline in all
three repetitions, producing .6917/.5000/.4089/0. The recovery safeguards work,
but this is a substantial quality loss compared with successful auto runs.

Prediction hashes are stable within every shape/budget/strategy group except
wider/25/auto, where deadline-dependent completion produces two different output
hashes. This is execution instability, not evidence that the estimator's random
seed changed. Timing variability itself is not independently replicated on
different machines or in a clean installed environment.

## What the phase measurements establish

Across the matched 18 pairs, routed costs 2.614 seconds more than auto on average.
At wider/50, the paired mean difference is 4.484 seconds (range 2.656 to 6.562).
At wider/25, a smaller wall-time difference masks different recovery outcomes;
wall time alone is therefore misleading.

Completed checkpoint publications average about .011 seconds in the available
normal-completion diagnostics. Routed wider/50 selection takes 8.00–8.56 seconds,
including histogram validation fits of 7.56–8.11 seconds, followed by final refits
of 7.78–8.89 seconds. These measured model phases are substantial costs. Single
candidate fits can exceed the nominal selection slice before the next budget
check; the outer worker deadline remains the enforcement boundary.

Parent elapsed time outside the instrumented load/profile/spawn/result phases
averages 3.653 seconds for routed versus 1.234 baseline and 1.241 auto. Worker
imports alone average approximately 2.44–2.56 seconds when results complete.
Code inspection shows that routed configuration loading imports `routing.py`,
which imports preprocessing and scikit-learn in the parent. This is a strong
startup-overhead hypothesis, but the residual also includes other setup/cleanup;
the study did not separately time or ablate that import. Do not attribute the
entire residual to one import or claim an optimization has already been achieved.

Process-tree RSS is **unavailable** because optional `psutil` is not installed.
All memory values are null, not zero. Memory acceptance also awaits official
limits. No dependency was installed or changed for the study.

## Recovery quality: measured capability versus expected benefit

There were four natural completed-baseline recoveries and two emergency fallbacks
among 54 runs. The normal/10 recovery preserved BA .7743 and minority recall
.6061. All three wider/25 recoveries preserved BA .5000 and minority recall zero;
their validity must not be described as useful minority recovery. This small
fixture set cannot estimate real assessment-time deadline probabilities.

A separate controlled demonstration fits a real normal-fixture baseline, forces
an optional-stage error, and reads the saved checkpoint through the same recovery
function with a simulated deadline reason. Recovered predictions exactly match
the completed baseline. Comparison against the actual emergency-majority fallback:

| Metric | Completed/recovered baseline | Emergency fallback |
|---|---:|---:|
| Ordinary accuracy | .8917 | .7250 |
| Balanced accuracy | .8406 | .5000 |
| Macro F1 | .8571 | .4203 |
| Minority recall | .7273 | .0000 |

This demonstrates that retaining completed results can preserve useful quality.
It is not a natural-timeout estimate or a quantified expected-score increase.
Existing real spawned-worker tests separately cover deadline termination, crash,
invalid final results, corrupted checkpoints, and emergency recovery behavior.

## Correctness and next decisions

The focused runtime/checkpoint/runner suite passed 35 tests in 76.23 seconds.
Mocked optional-success/error checks establish timing metadata is opt-in and
completed-checkpoint predictions are preserved. The initial real CLI timing
on/off test also passed byte-for-byte. That fixture was subsequently strengthened
from 16 to 96 rows with a nonempty-validation-score assertion, so the integrated
suite must verify the stronger version before declaring this batch complete.

Recommended next implementation order, behind controls and with paired tests:

1. Isolate parent import/configuration cost explicitly, then test lightweight
   parent configuration loading or deferred heavy imports. Preserve worker
   isolation, checkpoint publication, and prediction validity.
2. Add bounded per-candidate admission/refit-headroom evidence for expensive
   histogram work. Retain successful histogram gains rather than removing the
   candidate because it misses a short budget on this fixture.
3. Recheck 10/25/50-second paired quality and runtime, including recovery and
   emergency rates, after any overhead change. Distinguish deterministic model
   agreement from deadline-dependent completion differences.
4. Add process-tree memory measurement and official maximum-size/checker evidence
   once the environment and written limits are available.

Reject any optimization that removes recovery safeguards or improves normal
timing while worsening useful completed-model/recovery quality. No runtime
optimization, model promotion, production-default change, commit, push, or full
confirmation campaign was performed by this runtime workstream.
