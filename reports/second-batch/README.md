# Second targeted development batch

Starting checkpoint: `d770d8bbea25820dd0f4cbd231786e4427c7ca30`.
Independent recommendations: `reports/grade-evaluation/first-batch-review/REPORT.md`.
The working tree was clean before development. No commit or push is authorized.

The objective is useful predictive performance within a 60-second external
deadline. All scoring scenarios remain provisional. No model or runtime
alternative is promoted automatically, and default behavior is preserved.

## Responsibilities and experiment order

1. Runtime: optional lightweight parent configuration imports, isolated phase
   measurements and contemporary routed/lazy-routed/auto comparisons. Two shapes,
   internal budgets 10/25/50 seconds, three paired repetitions, with only the
   ambiguous wider 25-second block extended to five. The fixed inventory is 60
   runs. Accept an engineering speed claim only with at least one-second paired
   median saving and matching generous-budget predictions and selection. Report
   tight-budget quality and recovery separately.
2. Clustering: reproduce the eight known exceptions and compare indicators
   on/off crossed with current full-covariance geometry/K-means. Add informative
   missingness and new parameter settings. Preserve full-covariance restarts;
   never compare raw BIC across representations or select using reference scores.
3. Classification: replicate the original 12-case design on three new development
   seeds, retain three estimator families and comparable selection-time limits,
   and add separate rare multiclass interaction probes. Save input/split hashes,
   class supports and predictions for independent metric reconstruction.
4. Integration: correctness checks, default prediction compatibility against
   the checkpoint, and byte preservation of evaluator and historical evidence.

After the original 60-run import study still showed a missed optional refit,
the evaluator's conditional admission experiment was separately declared and
run: 16 serial calls (five wider/25 pairs, three wider/50 pairs). Both variants
use lazy imports. Its fixed cost multiplier and safety margin were frozen before
results. See admission-protocol.md and admission.md; the guard harms quality at
25 seconds and remains disabled. No follow-up tuning was performed.

Source ownership is separated by module. CPU-heavy studies run serially. Runtime
sources and configuration are frozen before benchmarking; offline harness changes
and resumed suffixes retain their own provenance and never overwrite evidence.
The two source ZIPs preserve the pre-admission import-study version and the later
admission-study version. Their differing hashes are intentional; neither archive
is a claim that all studies used the final source version.
All generated scratch files live under ignored `experiments/work/`.

These are inspected development cases, not unopened confirmation. Repeated
seeds and related synthetic mechanisms do not create independent family evidence.
Official scoring/checker, maximum-size and memory verification remain external
gaps. Regression estimators and existing improvements stay protected.

See runtime.md, clustering.md and classification.md for protocols and results;
SUMMARY.md records integration outcomes and the recommended next work.

## Optional controls

`--strategy routed --lazy-router-imports` enables lightweight parent imports.
Without that flag the previous import path is retained. Worker isolation and
checkpoint recovery remain active in both cases.

A router JSON override containing
`{"experimental": {"refit_admission": true}}` enables the separate admission
heuristic, which is classification-only and is **not recommended for promotion**.
It requires the original three-family selector: nondefault classification
weighting, holdouts, objective overrides and the categorical candidate are
rejected in combination. The CLI supplies the parent absolute deadline; direct
API calls without one report a local-budget estimate. Regression models do not
use this heuristic. The default is false.

The existing clustering representation/covariance controls and classification
holdout controls remain optional; this batch does not change their defaults.
