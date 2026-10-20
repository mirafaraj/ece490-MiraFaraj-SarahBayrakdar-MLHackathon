# Third batch: regression capacity protocol

Frozen before fitting. This offline development study tests histogram leaf capacity 15 versus 31. Ridge, Extra Trees, preprocessing, boosting iterations, minimum leaf size and candidate count remain unchanged. It is not confirmation or deadline evidence.

## Fixed design

18 case/seed combinations: sparse linear control, numeric interactions, mixed categorical nonlinear, missing product, heavy tail, heteroscedastic target × seeds 98003, 98017, 98029. Related mechanisms and repeated seeds are development probes, not independent task families. Each has 1,200 rows, 12 independent normal numeric features and one uniformly distributed five-level category. Exact equations and random generation order are fixed in `scripts/third_regression.py::case_data`. Missingness is independent 5% per numeric cell, except missing-product 20%. The linear control also has missingness and a nuisance category. Targets are generated before masking.

Outer 75/25 and inner 75/25 splits both use the case seed. Inner indices are subsets of outer training indices. Every cleaner, encoder, imputer and model is independently fitted on its fitting partition. Evaluation labels are absent from fitting and selection APIs.

Four candidates per partition: Ridge, Extra Trees, histogram-15, histogram-31. Each is fitted once inside and once on all outer-training rows: 18 × 4 × 2 = 144 fits. Histogram inner/final iterations remain 60/100; Extra Trees 40/96. The 96-tree final Extra Trees is the original 50-second baseline (same preprocessing, max depth 24, minimum leaf 1, seed and single thread). Baseline equivalence must additionally pass integration checks.

Two three-family portfolios replace only the histogram candidate. RMSE and MAE each select from inner predictions only, using existing simple-model exact-tie preference. Descriptive selection replay uses candidate order Extra Trees → linear → histogram and a shared 20-second cap, checking elapsed work before each candidate as production does. Failed or budget-excluded candidates do not enter selection; fit cost is retained even on errors. No extra folds, adaptive tuning, near-tie threshold, final-sized validation or confirmation cases are permitted. Offline exhaustive fitting is explicitly distinct from bounded production selection.

## Metrics and frozen decision gate

Save inputs, targets, split indices, references, predictions, SHA-256 values, timing, exceptions and full model metadata. Report RMSE, MAE, p90/p99 absolute residuals, errors on rows with/without missing values, dummy-relative errors, oracle gap, candidate exclusions and selection/refit time. Baseline is final Extra Trees; current/candidate are separately selected three-family policies. Source hashes are checked before/after study.

Nonlinear/mixed target subset is all five non-control mechanisms; each mechanism has equal weight. A declared objective warrants independent evaluation only if mean relative error reduction is at least 3%, positive on at least two of the three seeds, and mean linear-control deterioration is no more than 2%. Preserve conflicting RMSE/MAE alternatives. Explain severe individual regressions and whether fixed-candidate effects agree with selector effects. No default promotion follows this development gate.

The protocol above was frozen before execution. CLI runtime/deadline evidence is a separate parent-coordinated stage only for shortlisted alternatives.

## Completed development results

The single serialized run completed all **144/144 fits**, with zero exceptions, nonfinite outputs or candidate exclusions, in **93.23 seconds total offline work**. Four focused tests passed before the run. Python/package/platform versions and unchanged source hashes are recorded in `experiments/third-batch/regression.json`; independent arithmetic summaries are in `regression-summary.json`.

| Measure, mean across 18 cases | Original Extra Trees baseline | Current three-family selector | Capacity-31 selector |
|---|---:|---:|---:|
| RMSE | 3.47855 | 3.03815 | 3.06650 |
| MAE | 2.36507 | 1.99234 | 2.03671 |
| p90 absolute error | — | 4.29190 | 4.34044 |
| p99 absolute error | — | 11.14181 | 10.83894 |
| Dummy-relative RMSE | — | .56369 | .56901 |
| Dummy-relative MAE | — | .49052 | .50144 |

Means combine different synthetic target scales and are descriptive. The frozen acceptance gate uses within-case relative effects and equal family weights. Current RMSE/MAE selectors both reduce error against baseline by **11.91% / 15.53% mean relative**, versus **11.20% / 14.00%** for the challenger. Both objectives selected exactly the same family per case; conflicting objective outcomes did not arise here and both objectives remain supported.

| Mechanism | Challenger RMSE reduction vs current | Challenger MAE reduction vs current |
|---|---:|---:|
| Sparse linear control | 0.00% | 0.00% |
| Numeric interactions | -1.55% | -3.28% |
| Mixed categorical nonlinear | 0.00% | 0.00% |
| Missing product | -0.65% | -2.57% |
| Heavy tail | -0.69% | -2.08% |
| Heteroscedastic | -2.23% | -3.26% |
| Frozen nonlinear/mixed subset | **-1.02%** | **-2.24%** |

Negative reductions mean deterioration. All three seed-level target effects are negative for both objectives, so **both acceptance gates fail**. Linear predictions are identical in all three cases; the original selected Ridge gains are preserved: mean relative RMSE/MAE reduction versus Extra Trees **17.79% / 27.63%**.

Selection families do not change: **Ridge 3, histogram 12, Extra Trees 3**, under each capacity and objective. Selection regret relative to the evaluated outer-best candidate in its own three-family portfolio is zero in every case. Consequently this result isolates capacity rather than selector error. Fixed histogram-31 itself worsens mean relative RMSE **1.97%** and MAE **3.59%** across the 15 nonlinear/mixed cases. The largest fixed-candidate regression occurs in mixed-category seed 98003: RMSE **+9.47%**, MAE **+10.76%**; Extra Trees is correctly selected there, avoiding that regression. The largest selected-policy regression is numeric-interaction seed 98003: RMSE **+4.68%**, MAE **+6.56%**. Numeric-interaction seed 98029 does improve slightly (RMSE -.20%, MAE -2.04%), but does not offset the other seeds.

The p99 residual tail improves **2.16% mean relative**, while p90 deteriorates **1.13%**. This secondary tail result does not satisfy the declared RMSE/MAE objective and is not grounds for promotion. Mean MAE on missing rows worsens **2.27393→2.31299**, and on observed rows **1.63324→1.67066**.

## Cost, interpretation and next action

Median three-family selection work is **1.125→1.156 seconds**; median selected refit **.290→.352 seconds**. Fixed histogram inner fit/prediction median is **.1955→.2030 seconds**, final fit/prediction **.2810→.3285 seconds**. All candidates remain far below the descriptive 20-second selection cap, with zero exclusions. No recovery path is needed in these direct candidate fits; these are not subprocess crash-recovery or maximum-size deadline measurements. As neither quality gate passes, no additional CLI quality benchmark is warranted for this variant.

Additional histogram leaves did **not** produce a useful generalization gain in this controlled sample; larger capacity generally reduced useful regularization. Training residuals are not stored, so overfitting is a plausible mechanism, not a separately confirmed causal diagnosis. No model changes, extra leaves, new folds or tuning were added after seeing results. There is no independent repetition of these new fits; exact inputs, split indices, predictions and source hashes support replay, not an independent reproducibility claim.

Recommendation: **do not shortlist capacity 31 for independent score evaluation**. Keep the optional control off and retain existing Ridge/Extra Trees/histogram-15 alternatives. Preserve the linear gains and examine a separately approved representation improvement for nonlinear mixed features next; this batch gives no justification for final-sized validation or additional folds because measured selection regret is zero. Evidence remains restricted to six related synthetic mechanisms, three seeds and 1,200 rows, without an established official metric or independent confirmation.
