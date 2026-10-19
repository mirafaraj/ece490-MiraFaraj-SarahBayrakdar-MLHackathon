# Independent grade improvement evaluation

> Historical evaluation of the original frozen comparison. LATEST_EVALUATION.md
> evaluates the completed router and the new objective experiment under the
> user's current tradeoff-aware preference. Use STAGE_REVIEW.md for current phases.

Evaluated 19 October 2026. Scope: read-only source review and independent arithmetic on saved reports. No model fits, benchmarks, tests, installs, downloads, source edits, or messages to the implementation chat were performed by this evaluator. New artifacts are confined to this directory.

## Recommendation and immediate priorities

Treat the evaluator as a quality gate: produce a ranked improvement backlog, require paired evidence, and reject changes that only improve the chosen toy family. Do not promote all of `auto` as one unit. Classification, regression, and clustering need separate promotion decisions.

1. **Prepare metric-aligned classification ablations.** Existing evidence already reverses the headline conclusion under ordinary accuracy. Preserve both metrics; compare unweighted and balanced models once the official metric is known. This has greater expected value than tuning another classification hyperparameter against an uncertain objective.
2. **Extend clustering beyond the GMM fix in progress.** Mixed categorical clustering and curved geometry remain weak even if GMM initialization is repaired. Design numeric/category weighting and bounded graph-geometry ablations, with no blanket replacement of K-means. Leave the active GMM owner to finish and use its frozen result as a separate comparison.
3. **Target mixed/missing regression and selection reliability.** Current selection adds zero improvement on that family, and diabetes changes direction across splits. Compare narrowly scoped representations and a conservative repeat-holdout selector before adding a larger search.

The user's rubric allocates 85 total points equally across three task groups (85/3 each), profiling 10, and speed 5 conditional on useful quality. This guides effort allocation only. Official per-dataset metrics, normalization, group aggregation, profile schema, size limits, and assessment environment remain unavailable; no total grade or predicted point gain is computed here. A small timing gain cannot justify a demonstrated quality regression on the evidence available.

## Evidence and reproducibility

The historical frozen comparison is `experiments/broad-comparison.json`, source SHA-256 `81a8a622c85eca5d8286deefeb498cbe3dfe58dfb43cb3519e7214b1d22c51ef`, report SHA-256 `8563a97a8583422b856b554218143a6bbb66d58a34e9fbbb5606d3f94b506ac2`. It contains 48 runs (12 families, seeds 42 and 314, baseline/auto) with `source_unchanged=true`. The analyzer independently paired identical input hashes and reproduced **all aggregate fields** of `experiments/comparison_summary.json`, rather than trusting its win labels.

`analyze.py` is a reusable standard-library-only artifact reader; it never imports production or ML libraries, generates datasets, or executes another process. It rejects duplicate/missing strategies, mismatched paired input hashes, and inconsistent summary aggregates. It writes `scorecard.json` with raw paired deltas, reversals, per-family results, wall times, and observed input shapes.

```powershell
.\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py experiments/broad-comparison.json --summary experiments/comparison_summary.json --output reports/grade-evaluation/scorecard.json
```

Run it against another report using a **new output filename**. The optional `--summary` is for a summary of that same source, and must be omitted if none exists. The script expects the existing evaluator's paired baseline/auto schema. It validates saved metric arithmetic and provenance, not predictions or mathematical correctness of the saved metrics themselves. It intentionally does not derive an official score or compare unrelated RMSE scales.

Current source is moving: `clustering.py` already contains complementary generic/baseline-centroid GMM starts absent from the historical result. The active implementation chat owns its experiments. The numbers below describe the frozen source, not proof of current GMM performance. The earlier low-budget tree-count issue is fixed in current `supervised.py`; do not reopen it as an unresolved bug merely because an older cross-review mentions it.

## Findings that change improvement priorities

### Classification: apparent improvement depends on the metric

Across eight paired classification cases, mean balanced accuracy increases **7.671 percentage points**, while mean ordinary accuracy decreases **2.991 percentage points**. The two imbalanced cases reverse direction:

| Seed | Balanced accuracy change | Ordinary accuracy change |
|---|---:|---:|
| 42 | +29.516 pp | -12.234 pp |
| 314 | +27.604 pp | -14.894 pp |

On seed 314, ordinary accuracy changes 0.93085 to 0.78191 while balanced accuracy changes 0.50000 to 0.77604. Both baseline and candidate classification estimators use balanced class weights; `supervised.py` also selects candidates by balanced accuracy. This aligns with equal class recall, but it is not evidence of maximum official classification quality if the scorer uses accuracy. An unweighted candidate has not been measured here, so its improvement remains a hypothesis.

The other families show balanced-accuracy mean changes of about +0.860 pp on breast cancer, +1.266 pp on high cardinality, and zero on wine. The imbalanced family dominates the headline mean. With only two splits and one already-perfect small family, no broad dominance claim is justified.

### Regression: real gains, concentrated in a synthetic linear family

Use mean **per-case relative RMSE reduction**, not a mean of raw RMSE improvements across target scales.

| Family | Mean relative RMSE reduction across two seeds | Interpretation |
|---|---:|---|
| Linear | 87.343% | Strong family-specific Ridge advantage |
| Nonlinear | 25.485% | Histogram boosting adds useful quality |
| Diabetes | 1.882% | Seed 42 worsens 3.073%; seed 314 improves 6.838% |
| Mixed/missing | 0.000% | Extra Trees selected on both seeds; no candidate gain |

Overall mean relative RMSE reduction is **28.678%**; excluding the linear family it is **9.123%**. These are useful diagnostics, not normalized contest scores. Mixed/missing seed RMSEs differ materially (3.4817 versus 1.7402), so inspect residual tails and missing-row subgroups before concluding the family is solved. Targets were generated before feature corruption; some error is irreducible from hidden values and cannot be repaired by fitting harder.

Selection uses one training-only 25% holdout, at most 4,000 sampled rows. Extra Trees is scored with 40 trees then refitted with 96; histogram boosting uses 60 versus 100 iterations. This is a runtime tradeoff, not an established correctness bug, but the ranking is an approximation to final estimators. The diabetes reversal is consistent with selection uncertainty; it does not by itself identify its cause. No evaluation-label leakage was found in the reviewed preprocessing/selection path.

### Clustering: two untouched weaknesses beyond GMM initialization

| Family | Mean baseline ARI | Mean historical auto ARI |
|---|---:|---:|
| Noisy blobs | 0.9980 | 0.8149 |
| Anisotropic | 0.8847 | 0.9977 |
| Moons | 0.4662 | 0.4662 |
| Mixed clusters | 0.4896 | 0.4896 |

The historical seed-42 noisy-blobs regression (1.0000 to 0.6337) is the active GMM repair's responsibility. Independent next work should target mixed clusters and curved shapes. Current geometry selection excludes tables with categorical columns entirely. Mixed tables therefore use scaled numeric plus one-hot Euclidean K-means without a measured category/numeric block weighting policy. The category is informative but intentionally corrupted; over-weighting it may also hurt. Numeric missingness indicators add another geometric contribution. These are representation hypotheses to test, not confirmed causes.

Moons is a deliberate weakness of centroid/elliptical geometry. A graph candidate can help that shape, but indiscriminate selection by silhouette or BIC can prefer the wrong semantic partition. ARI references belong only in the evaluator; production routing must use training features and a frozen, general rule. `_occupy_groups` repairs empty groups to satisfy the provisional exact-k checker; record those repairs because valid cardinality alone says nothing about useful quality. Whether exact occupancy is officially required remains unconfirmed.

### Profiling: broad provisional correctness, uncertain official coverage

The frozen report's 48 valid runs passed the evaluator's independent raw-CSV checks for counts, types, missingness, constants, exact duplicate columns, numeric min/max/mean/median/population standard deviation, and target summaries/distributions. `test_profiling.py` also covers all-missing data, differing missing positions, input preservation, and target-length mismatch. This is stronger evidence than merely checking for nonempty JSON.

The schema is explicitly `provisional-1`. Feature category frequency distributions, quantiles beyond median, duplicate row counts, outlier counts, correlations, and inferred semantic/date types are not emitted. **These are coverage gaps only if the official profile rubric requires them.** Avoid adding every imaginable statistic at the cost of runtime. Current semantics deserve explicit mapping: training data only; target and declared ID excluded from feature totals; constants ignore missing values; duplicates require equal dtype and missing positions; standard deviation uses ddof=0; CSV default NA tokens can turn literal category strings into missing values. An official checker may use different definitions. Production numeric summaries filter infinities, but the CSV contract rejects nonfinite numeric input; do not imply such inputs are covered in the benchmark.

### Speed and fallbacks: small-table evidence only

All 48 historical runs are valid with zero deadline fallbacks. Full CLI wall times range **3.2153–7.1647 s**, median **4.15245 s**. However the largest observed training table has **660 rows**, and the largest input width is **30 features**. This provides no maximum-size or memory evidence. The 50-second internal budget, one-thread policy, process deadline, and existing worker-termination tests are sensible safeguards; they do not establish 60-second full-process success for large inputs.

Loading, profiling, process startup/serialization, fallback, and output are outside the fitted model's interruption boundary. The dense cap guards particular conversions, not all allocations; selection can skip above 100,000 rows or 1,000 features, but final training can still be expensive. A deadline fallback may be valid while providing majority/mean/arbitrary group predictions and losing model-quality points. Report fallback quality and full wall time together; do not count valid output as quality success.

## Ranked experiment backlog with concrete gates

The following thresholds are **our proposed engineering acceptance gates**, not contest rules. Until metrics arrive, run the specified secondary metrics and preserve alternatives rather than declaring a universal winner. No experiments in this table were run by this evaluator.

| Priority / change | Exact ablations and cases | Measurements and proposed gate |
|---|---|---|
| P0: metric-aligned classification | Compare current balanced candidates against identical Extra Trees, logistic, histogram candidates with `class_weight=None`. First vary weighting only, then selector objective only (accuracy versus balanced accuracy). Use imbalanced prevalences 50/50, 90/10, 97/3, the current breast-cancer/wine/high-cardinality families, and at least one multiclass imbalance case. | Report accuracy, balanced accuracy, macro F1, minority recall, and per-class counts. Promote under the official metric only: mean paired gain at least 0.5 pp, no family mean loss over 1 pp, no single-case loss over 3 pp, zero invalid runs/fallbacks on ordinary-size cases. Without official metric retain both policies; never call the observed reversal a win unconditionally. |
| P1: mixed clustering representation | Keep K-means/k/seed fixed; numeric standard scaling versus robust scaling; numeric/category block weights 0.5, 1, 2 (one factor at a time); indicators enabled versus excluded from clustering distances. Include mixed_clusters, numeric-only controls, category-only signal, irrelevant categories, missingness levels 0/10/30%, and rescaled numeric features. | External ARI for diagnostics, wall time, cluster sizes, repair count. Select a global policy using development cases, never per-case reference labels. Require target-family mean ARI +0.03, no control-family mean loss over 0.02, no single-case drop over 0.05, zero invalid/fallbacks; reject a policy that wins only when categories directly encode labels. |
| P1: bounded curved-shape clustering | Compare K-means with fixed-k spectral nearest-neighbor clustering for only small numeric tables (proposed eligibility n<=2,000, transformed dimensions<=16), neighbors 10 and 20 as development alternatives. Freeze eligibility and any unsupervised selector before confirmation. Cases: moons, circles, blobs, anisotropic, overlapping blobs, irrelevant-feature controls. | Require moons/circles mean ARI +0.10 and no other family mean loss over 0.02; additional median CLI cost <=3 s on eligible small cases; zero invalid/fallbacks. A selector must outperform retained K-means controls; an oracle best-of with reference ARI is not deployable evidence. Keep as opt-in if safe routing is unproven. |
| P1: mixed/missing regression | Keep selector/splits fixed; compare existing Extra Trees to histogram boosting on bounded dense one-hot input; then a compact histogram variant (`max_leaf_nodes` 15 versus 31; `min_samples_leaf` current versus 5) one factor at a time. Compare numeric missing indicators on/off separately. Include mixed_missing at 0/10/30% missingness, nonlinear, linear, diabetes, and high-cardinality controls. | Report relative RMSE, R2, MAE, residual 90th/99th percentiles, missing versus observed-row error, selection/refit times. Require mixed-family mean relative RMSE reduction >=5%, positive median gain, no family mean degradation >2%, no case degradation >5%, zero invalid/fallbacks. Do not tune on the external residual labels; they diagnose frozen candidates. |
| P2: selection stability | Compare one holdout to two deterministic training-only holdouts at the same total selection budget; compare current cheap proxy scoring against equal-budget final-sized candidate scoring; freeze either a 1% minimum advantage rule or train-only top-two blending for regression as separate candidates. Cases: diabetes, mixed_missing, weak-signal regression, linear/nonlinear controls. | Report winner frequency, score margins, outer-test metric, selection/refit cost; require mean relative RMSE improvement >=1%, fewer >5% regressions than current selector, no loss of linear advantage beyond 2% relative RMSE, zero new fallback cases. Avoid spending scarce time on repeat validation when estimated refit cannot finish. |
| P2: large-case survival and graceful quality | After official limits arrive, cover maximum rows, maximum width, many categories, large test sets, constant data, and combined worst permitted shape on official hardware/runtime. Until then label any proposed 10k/100k-row probes exploratory, not compliance. Compare current path to reduced-tree/sample-fit policy under a fixed external watchdog. | Record external wall time, peak process-tree RSS, parse/profile/model/output times, exit/artifact validity, fallback rate, and task quality versus majority/mean/random-group diagnostics. Proposed gate: every permitted-shape case valid <60 s; aim <=55 s headroom; no deadline fallback on expected normal cases; never trade a useful model for a fast dummy merely to earn speed credit. Memory gate awaits the published limit. |
| P0 at kickoff: profile/contract mapping | Build a field-by-field table from official schema/definitions to current output, including std ddof, missing tokens, type inference, IDs, target scope, duplicates, ordering, output file names, cluster protocol. Use exact hand-counted edge fixtures and the official practice checker. | All required official fields/semantics and nine practice datasets pass; no guessed extra fields if the checker rejects them; profile time included in <60 s full run. Use independent expectations rather than asserting the production function's output against itself. |

## Evaluation protocol and scheduling

1. **During the active timed GMM work:** this evaluator performs artifact analysis only. Do not run ML fits, pytest, benchmark subprocesses, install packages, edit shared production files, or change existing experiments. This report and analyzer are the concrete parallel deliverable.
2. **After that owner finishes:** nominate one experiment owner per production area and one evaluator. Record exact source/candidate configuration and input hashes; run timed evaluations serially on an otherwise idle machine. Reuse identical input files for baseline and candidate; alternate/randomize strategy order across repeats to diagnose startup/cache effects. Keep quality comparisons paired and runtime summaries per task/shape.
3. **Separate development from confirmation:** seeds 42 and 314 are now exposed and may guide development. Any already-opened 2027 results are also development. Reading a confirmation result and then tuning a policy means it cannot confirm that tuned policy. Real-data new splits are useful, but not new independent datasets.
4. **Freeze candidates, then create fresh unopened confirmation:** choose several seed values only after all candidates, eligibility rules, thresholds, and acceptance gates are fixed; do not enumerate or inspect their results while choosing candidates. Include unseen generator settings/families and independent real datasets when available. Store source/input/config hashes and a manifest before the run. The evaluator opens results only once the frozen run finishes; it may approve or reject the candidate, not retune it on those results. Retuning starts another development cycle and requires a new unopened final set.
5. **Decision report:** for each task, publish per-family wins/ties/losses, worst regression, paired mean/median, invalids/fallbacks, external p50/max wall time and memory, and independent family count. Use bootstrap intervals by family only if enough independent families exist; two seeds of one table must not be presented as two independent datasets. Keep classification accuracy and balanced accuracy alongside each other until official alignment is settled. Never combine raw RMSE, ARI, and accuracy into a pseudo-grade.

## Work that specifically waits for official information

- Official metric and normalization determine classification weighting/thresholds, regression loss/target transforms, clustering objective, and the actual per-dataset aggregation. Preparing ablations can start before that; selecting the scoring winner cannot.
- Official feature typing, missing-value treatment, exact profiling fields, cluster-count protocol, I/O, environment, allowed dependencies and network policy determine compatibility. Keep current provisional contract clearly labeled.
- Official maximum dimensions, memory and hardware determine stress cases and budget routing. Current 660-row/30-feature maxima must not be used as evidence of a maximum-size pass.

The next useful evaluator output is an evidence-backed **promote / reject / insufficient evidence** decision for a frozen candidate against these gates. This report does not authorize changing production defaults based solely on historical development averages.

## Ready-to-run targeted commands for the later experiment owner

These commands are **deferred until the active timed work ends and source is frozen**. They run actual ML fits and are not part of this evaluator's artifact-only work. They exercise the currently supported baseline/auto strategies; implementing the ablations above requires a separately owned source change and a new frozen fingerprint. No invented CLI flags are needed. Seeds 42/314 below are explicitly reused development evidence, not fresh confirmation.

```powershell
# Classification metric reversal and controls (16 CLI runs).
.\.venv312\Scripts\python.exe -u scripts/evaluate.py --suite broad --seeds 42 314 --strategies baseline auto --cases imbalanced breast_cancer wine high_cardinality --output reports/grade-evaluation/future-classification-development.json
# Mixed regression and selection controls (16 CLI runs).
.\.venv312\Scripts\python.exe -u scripts/evaluate.py --suite broad --seeds 42 314 --strategies baseline auto --cases mixed_missing diabetes linear nonlinear --output reports/grade-evaluation/future-regression-development.json
# Curved/mixed clustering with an anisotropic control (12 CLI runs).
.\.venv312\Scripts\python.exe -u scripts/evaluate.py --suite broad --seeds 42 314 --strategies baseline auto --cases moons mixed_clusters anisotropic --output reports/grade-evaluation/future-clustering-development.json
# Recompute a scorecard after each report exists; choose a new filename per source freeze.
.\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py reports/grade-evaluation/future-classification-development.json --output reports/grade-evaluation/future-classification-scorecard.json
```

Run each command sequentially and change output names for each frozen candidate; the benchmark also creates temporary inputs under its existing `experiments/work` directory. These are focused measurements, not a substitute for the broader family/settings coverage or unopened final confirmation required above.
