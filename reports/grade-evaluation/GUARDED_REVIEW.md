# Accuracy-preserving candidate assessment

> Historical strict assessment. The user subsequently requested retaining useful
> improvements and reviewing tradeoffs instead of automatic rejection for an
> ordinary-accuracy loss. See LATEST_EVALUATION.md and ACCURACY_POLICY.md for the
> current decision policy. The strict gate below remains an optional diagnostic.

The evaluator now makes a concrete decision: **reject historical `auto`** for accuracy-preserving promotion. Across eight paired classification cases, ordinary accuracy averages -2.991 percentage points and the worst case is -14.894 pp. The imbalanced cases lose 12.234 pp (seed 42) and 14.894 pp (seed 314), although balanced accuracy increases. Higher balanced accuracy cannot override the user's no-drop requirement.

`guarded-historical-scorecard.json` is newly generated from the original frozen 48-run artifact. Original `scorecard.json` is preserved. All legacy per-partition/task aggregates still independently reproduce the existing comparison summary. The new machine-readable assessment lists per-case, per-family, and overall ordinary accuracy, balanced accuracy as a secondary metric, candidate failures, evidence gaps, missing coverage, and ranked improvement advice grounded in the paired cases.

## What the strict gate enforces

- Any finite in-range ordinary accuracy loss below -1e-12 on a valid matched-input pair rejects, even if overall/family averages improve or unrelated metadata is missing. The tolerance is numerical precision, not a permitted performance tradeoff.
- Invalid outputs, recorded errors, introduced emergency fallbacks, and observed deadline failures reject. Missing/nonfinite metrics, absent fallback/exit/timing data, missing pairs/seed coverage, bad input/source provenance, incomplete streaming reports, and missing classification coverage cannot pass. Existing candidate recovery metadata is retained to distinguish baseline recovery from an emergency fallback; those are not automatically treated as equivalent.
- Other task metrics remain separate diagnostics. A usable regression/clustering row does not mean that its quality passed a promotion gate.
- `passed` means the observed development accuracy gate passed only. Actual promotion remains `insufficient_evidence` until official metric/profile alignment, frozen fresh unopened confirmation, and maximum-size evidence exist. No code can promise hidden-set accuracy from a few visible cases.

## Use on the next completed frozen result

```powershell
.\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py experiments/broad-comparison.json --summary experiments/comparison_summary.json --output reports/grade-evaluation/guarded-historical-scorecard.json --enforce-accuracy-gate
# After the implementation owner supplies a completed baseline/routed CLI report:
# .\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py <completed-report.json> --candidate-strategy routed --output reports/grade-evaluation/guarded-routed-scorecard.json --enforce-accuracy-gate
```

Enforcement returns 2 for rejection or insufficient evidence; report-only mode returns 0 while still writing the negative decision. `--summary` is restricted to the original baseline/auto aggregate schema. Legacy aggregate names containing `auto` denote the requested candidate; `candidate_strategy` and `gate_assessment` identify it explicitly. Reports with additional strategies ignore unrelated strategies. A missing requested strategy cannot pass.

Current source has an objective-aware `routed` strategy and supports unweighted fits when the objective is ordinary accuracy. This is a useful implementation capability, **not measured proof of no drop**. The other chat is integrating and validating it; no completed frozen routed CLI artifact was available for this assessment. The separate selector study uses a diagnostic schema and a 120-second model budget, so it is not a 60-second production CLI pass and is not fed into this gate. Historical `auto` results must not be misattributed to `routed`.

## Relevant next improvements

1. Preserve baseline classification until a frozen candidate passes every measured ordinary-accuracy case. Configure training-only candidate selection for ordinary accuracy and compare unweighted versus balanced fits. Do not choose candidate routes using external evaluation labels.
2. Improve mixed/curved clustering as a separate task route while the existing GMM owner finishes. Historical ARI remains roughly 0.49 for mixed tables and 0.47 for moons, with no `auto` gain; those are concrete weaknesses beyond initialization.
3. Target mixed/missing regression and selection stability. That family ties both seeds; diabetes changes direction. The overall 28.68% mean relative RMSE reduction falls to 9.12% excluding the synthetic linear family. Use family-level and worst-case comparisons so the large linear gain cannot conceal regressions.

The detailed candidate ablations and deferred experiments remain in `REPORT.md`. Its earlier proposed aggregate classification thresholds are superseded by this strict per-case no-drop gate wherever they conflict. All benchmark/fitting commands remain deferred to the implementation owner after its active timed work ends; this evaluator runs only standard-library artifact checks.

Validation: focused `unittest` fixtures cover mean masking, numerical tolerance, invalid/missing/nonfinite metrics, arbitrary candidate strategy, subset tasks, paired input mismatch, incomplete provenance/seeds, fallback regression, deadline checks, duplicate rows, known loss with unrelated missing metadata, malformed artifacts, and CLI enforcement/report-only behavior. No production files or tests were changed.
