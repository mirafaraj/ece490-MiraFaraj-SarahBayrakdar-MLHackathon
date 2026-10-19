# Independent review, expanded evaluation, and model experiments

## Outcome

Three independent reviews were completed, followed by cross-review of both new model candidates.
Correctness fixes and regression tests are implemented. The broadened benchmark contains
12 families, two seeds, and two strategies: **48 actual CLI runs, all valid, zero fallbacks**.
CLI wall times ranged from about 3.2 to 7.2 seconds on these small tables. No official contest
scores or worst-case runtime claim can be made from this suite.

Final current-source verification: **63 tests passed**, plus **6/6 CLI smoke runs** across
both strategies after the low-budget fix. Both benchmark reports confirm stable source during
their respective runs. Two expected convergence warnings exercise constant-table cluster repair.

## Review fixes

- Classification CSV labels preserve lexical identity, including `01` versus `1`.
- Inferred categorical test columns use training parsing semantics.
- Numeric imputation retains missingness indicators; real categories cannot collide with missing tokens.
- Failed reruns clear previous generated artifacts; output writes are staged before publication.
- Evaluator checks the declared cluster count and preserves fractional/lexical classification labels.
- Profile calculations are independently checked against raw input values.
- Forced-worker tests exercise termination and marked deadline fallbacks.
- Cross-review found a low-budget candidate fitting 96 trees when baseline used 32. This was
  corrected after the frozen comparison, with a regression assertion. At the benchmark's 50-second
  budget, the tree count remains 96, so the model settings used for the comparison are unchanged.

No benchmark reference labels enter candidate selection. Supervised preprocessing is fitted
inside its training-only split and refitted on all training data after selection. Clustering
uses training-data density criteria, without reference labels.

## Paired results

Wins use balanced accuracy for classification, lower RMSE for regression, and ARI for clustering.
Each row below summarizes four families. These are provisional objectives, not official scores.

| Partition | Task | Wins / ties / losses | Average quality change |
|---|---|---|---|
| Development, seed 42 | Classification | 2 / 2 / 0 | Balanced accuracy +7.81 percentage points |
| Development, seed 42 | Regression | 2 / 1 / 1 | Mean relative RMSE reduction 25.81% |
| Development, seed 42 | Clustering | 1 / 2 / 1 | ARI -0.0494 |
| Confirmation, seed 314 | Classification | 2 / 2 / 0 | Balanced accuracy +7.53 percentage points |
| Confirmation, seed 314 | Regression | 3 / 1 / 0 | Mean relative RMSE reduction 31.54% |
| Confirmation, seed 314 | Clustering | 1 / 3 / 0 | ARI +0.0144 |

The regression statistic averages each case's fractional RMSE reduction, rather than averaging
raw RMSE across incompatible target scales. Strong gains on synthetic linear data contribute
substantially to that average.

Important tradeoffs:

- Confirmation imbalanced classification: balanced accuracy improves 0.5000 to 0.7760, while
  ordinary accuracy falls 0.9309 to 0.7819. The official metric must determine which behavior is useful.
- Development noisy blobs: clustering ARI falls 1.0000 to 0.6337. Better density fit can produce worse
  reference-label agreement; this candidate is not a safe universal clustering default.
- Development diabetes: RMSE worsens 51.73 to 53.32; confirmation improves 61.69 to 57.47.
  One internal holdout has selection uncertainty.
- Confirmation synthetic linear regression: RMSE improves 104.35 to 10.84; nonlinear regression
  improves 2.18 to 1.53. Mixed/missing regression ties on both seeds.
- Moons remain a weakness; both strategies score roughly 0.46–0.47 ARI.

## Default decision

**Retain `--strategy baseline` as the default.** The supervised candidate is useful and can be
selected explicitly with `--strategy auto`, but its objective is provisional. The clustering
candidate has a substantial observed regression and remains experimental. No candidate was
retuned after inspecting confirmation results.

The baseline here includes the review's correctness fixes. The historical three-case report
`experiments/baseline.json` remains unchanged and used the earlier preprocessing version.

## Reproducibility and evidence

- `experiments/broad-comparison.json`: all 48 per-case results, metrics, profile checks, timing,
  strategy, selected models, input/source hashes, and source-stability check.
- `experiments/comparison_summary.json`: paired wins/losses and per-task aggregates.
- `experiments/source-81a8a622c85eca5d.zip`: exact source snapshot used by the frozen comparison.
- `experiments/post-review-smoke.json`: current-source CLI checks after the isolated low-budget fix.
- `reports/review_*.md`: independent review findings and reproductions.
- `reports/candidate_cross_review_*.md`: cross-reviews of new candidates.
- `reports/supervised_experiments.md` and `reports/clustering_experiments.md`: independent development
  experiments, including limitations and tuning history.

Seed 42 was development; seed 314 was confirmation. Synthetic samples differ across seeds;
real datasets use different splits of the same source tables, so these are not independent
new real datasets. Family design was visible during development, and this remains a small suite.
Tests ran on Windows Python 3.12.14 with pinned libraries in the existing verified environment.

## Still outstanding

Official contract/metrics/checker and nine practice datasets; published maximum-size and memory
stress tests; completely clean installation and remote CI; additional independent real datasets
and split seeds; public GitHub repository, team names, and final submission commit. Candidate
fits are bounded heuristically; the outer worker deadline remains the hard interruption mechanism.
