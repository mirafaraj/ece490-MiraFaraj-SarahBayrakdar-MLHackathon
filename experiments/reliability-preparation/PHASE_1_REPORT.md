# Diagnosis stopped at the specified non-forest cause

| Phase | Allocation / actual status |
|---|---|
| Checkpoint | Complete: 4c76b2fd83ad2db7f6c427f839858a2d135a05c2 |
| Diagnosis | About 10 minutes; nine serial traced calls; stopped |
| Initial repair | Not started; maximum 4h if subsequently authorized |
| Profiles / submission / validation | Not run because the stop condition was reached |
| Freeze and final verification | Not started; cutoff Oct 21 15:00 Lebanon time |

Submission deadline is Oct 21 23:00 Asia/Beirut (20:00 UTC). At the end of
diagnosis, about 23h25 remained. Reserve the final eight hours; the freeze cutoff
is 15:00 Lebanon time, not a newly inferred organizer deadline. No frozen
candidate exists yet.

## Nine-case timing breakdown

All times seconds, diagnostic tracing enabled. Measurements include tracing
overhead and are not clean performance benchmarks. Forest figures marked `>`
are conservative lower bounds from completed tree events inside an interrupted
forest call; actual total forest time is unavailable. Hist binning never returned.

| Task / local shape / strategy | Load | Profile | Forest fit | State when worker stopped | Validation | Output write | Full process |
|---|---:|---:|---:|---|---:|---:|---:|
| Classification 100k x50 baseline | 1.42 | .66 | 25.47 | Finished, trained predictions transferred | 7.49 | .06 | 45.83 |
| Classification 100k x50 routed | 1.38 | .69 | 26.89 | Trained predictions existed, checkpoint validation interrupted | 3.73 parent; worker incomplete | .06, then rejected | 51.42 |
| Regression 100k x50 baseline | 1.42 | .64 | >35.10 | Forest interrupted, no trained predictions | <.01 | .06 | 47.50 |
| Regression 100k x50 auto | 1.45 | .64 | 1.69 selection + >29.32 final | Final forest interrupted | <.01 | .08 | 47.50 |
| Regression 100k x50 routed | 1.52 | .70 | >29.59 | Baseline forest interrupted, no checkpoint | <.01 | .11 | 47.70 |
| Regression 5k x500 baseline | 1.64 | 1.61 | >32.20 | Forest interrupted | <.01 | .25 | 47.64 |
| Regression 5k x500 auto | 1.56 | 1.64 | 15.86 selection + >13.64 final | Final forest interrupted | <.01 | .25 | 47.64 |
| Regression 5k x500 routed | 1.59 | 1.61 | >26.70 | Baseline forest interrupted, no checkpoint | <.01 | .24 | 47.84 |
| Classification 5k x500 auto | 1.92 | 1.72 | .88 selection | Histogram binning interrupted; no final fit | .11 | .19 | 47.70 |

Additional measured stages (nested timings must not be added together):

- Parent data imports: 1.28-1.50s; routed parent model/configuration imports:
  4.45-4.83s where output metadata survives. The invalid routed classification
  run has no surviving parent phase metadata.
- Worker process-start to model-dispatch entry: 6.05-6.78s, including spawn
  bootstrap/unpickling and imports. Within the worker function, model imports
  took 4.45-4.94s. Separating OS startup from unpickling is unavailable.
- Large classification baseline preparation before forest fit: about 1.92s;
  routed's fitted preparation inside `_fit`: about 1.58s, plus .66s routing.
- Large regression final preparation before forest fit: 1.44-1.75s. Wide
  regression final preparation: 2.63-3.77s, with routed planning 2.13s.
- Large regression auto completed selection before final refit: about 4.89s,
  including preprocessing/validation predictions. Wide regression auto spent
  about 19.80s before final refit and skipped further candidates after its cap.
- Wide classification auto reached histogram after about 8.34s of selection.
  Its forest estimator fit took .875s and linear estimator fit .078s;
  preprocessing accounts for much of the rest. Histogram preparation then took
  2.66s. The histogram/binning call remained active for roughly 24s until worker
  termination; its exact completed duration is unavailable.
- Successful baseline result transfer plus parent validation: 3.797s; validation
  itself was about 3.74s per call and occurred twice (7.49s total). Serialization
  of the worker payload and OS transfer were not separately instrumented.
- Recovery/result-validation phase after regression termination: .031-.063s.
  No completed routed checkpoint was recovered in any of these nine calls.

## Cause and stop decision

All six regression workers stopped inside forest fitting. Routed classification
finished forest predictions but was killed during validation for checkpoint
publication. It therefore had no complete atomic checkpoint to recover. Baseline
classification was valid on this repeat but exceeded 45s; its historical invalid
result is timing-sensitive. Large classification validation uses membership
against the full training-label vector, which adds measurable cost. Output
writing itself was small; its error message names the final budget check, not
necessarily the operation that consumed the budget.

**The ninth case violates the forest-only assumption:** wide auto classification
was killed inside `HistGradientBoostingClassifier.fit` -> `_BinMapper.fit`.
Its validation forest and linear predictions had completed, but they were not
final full-training predictions or published recovery checkpoints. Those
completed candidate scores cannot be treated as a usable final prediction file.

This activates the explicit instruction to stop when the principal cause is
something other than forest fitting. No production fix, first bounded preset,
linear-recovery trial, strategy addition, README change or further benchmark
was performed. The fixed forest may still help regression, but it does not
by itself explain or repair the histogram selection stall.

## Verification status and preserved state

- Nine diagnosed calls: eight CLI successes, including seven emergency outputs;
  one invalid routed classification output. All nine exceeded 45s; maximum
  51.422s. None exceeded 60s or triggered the external guard.
- Full output usefulness, full schema correctness and independent row-level
  validity have not been re-audited for this diagnostic subset. Existing CLI
  validation still ran. Do not interpret emergency output as useful learning.
- No new correctness suite or 108-run equivalence campaign was run. All original
  source/configuration hashes match, all 211 historical protected files match,
  and there are no tracked modifications. This is preservation evidence, not a
  new end-to-end prediction-equivalence claim.
- No production files changed. New files exist only below this report's folder;
  the exact inventory is `NEW_FILES.json`. Scripts `profile_reference.py` and
  `verify.py` are unexecuted preparation, not completed validation.
- Dependency lock hash and original source/config/environment records are in
  `phase-1-status.json`; inputs were checked against their original hashes before
  each diagnosis. Trace instrumentation is diagnostic-only via PYTHONPATH and is
  not installed or enabled for ordinary CLI use.
- No freeze hashes or submission strategy command exist yet. Submission remains
  unimplemented and non-default; CLI/API default remains baseline. The intended
  future explicit command remains an implementation assumption, not official.
- Announcement is the supplied specification. Metric formulae, concrete size
  limits, exact machine-readable contract and CPU hardware remain unspecified.
- No commits or pushes occurred during this phase. The only authorized checkpoint
  remains `4c76b2fd83ad2db7f6c427f839858a2d135a05c2`.

Await user instructions on extending the narrow repair scope to the histogram
selection stall; do not silently continue the forest-only repair or later phases.
