# Independent clustering experiment

## Recommendation

Keep baseline K-means as the default. The geometry strategy is opt-in: eight small development cases with seed 42 showed seven ties and one improvement. That is useful evidence for covariance-aware clustering, but not sufficient evidence of a general improvement on unfamiliar datasets. No confirmation314 data were generated or inspected by this agent.

## Candidate

`src/unknown_dataset/clustering.py` exposes `fit_predict_clustering` with the common task/model API. It fits the existing training-only cleaner and scaler, then runs the existing K-means/MiniBatch K-means settings. Only numeric tables with 1–16 retained numeric columns, at most 12 clusters, 40–20,000 rows (and at least 10 rows per cluster), and at least 12 seconds remaining are eligible for covariance comparison. Mixed categorical tables retain baseline behavior.

On at most 2,000 deterministically sampled training rows, compare spherical and full Gaussian mixtures. Each uses two initializations, at most 80 EM iterations, and covariance regularization. Select full covariance only when its BIC improves by more than `max(10, 5% of absolute spherical BIC)`. BIC uses no reference labels, but its density-fit objective need not correspond to the groups intended by private scoring. Failed/nonconverged mixture candidates retain baseline predictions. Empty groups are repaired deterministically to honor required k.

The outer model worker still owns the hard deadline. Candidate loops check time between fits; a running EM fit cannot be interrupted internally. Covariance dimension, samples, restarts, and iterations bound its work.

## Development comparison

All cases have 600 rows. Features were generated with NumPy/scikit-learn using seed 42. ARI is used only for offline diagnostics. Timings are in-process model calls and exclude interpreter/CLI startup. Baseline first-call timing contains additional warm-up overhead, so apparent speedups are not claims of better speed.

| Case | Baseline ARI | Geometry ARI | Geometry seconds |
|---|---:|---:|---:|
| Spherical blobs | 1.000000 | 1.000000 | 0.0394 |
| Transformed anisotropic blobs | 1.000000 | 1.000000 | 0.0373 |
| Unequal variances and group sizes | 0.858891 | 0.858891 | 0.0435 |
| Moons | 0.466053 | 0.466053 | 0.0468 |
| Nested circles | -0.001627 | -0.001627 | 0.0457 |
| Ten irrelevant features plus informative category | 1.000000 | 1.000000 | 0.0528 |
| Outlier contamination | 0.965222 | 0.965222 | 0.0535 |
| Correlated elliptical groups | 0.499714 | 1.000000 | 0.0649 |

The correlated ellipse fixture uses centers `[-2,2]`, `[0,0]`, `[2,-2]`, shared covariance `[[2,1.95],[1.95,2]]`, and 200 rows per group. It specifically probes a geometry that spherical K-means can split incorrectly. It was added during development and must not be treated as an untouched confirmation case.

An initial one-initialization mixture configuration produced a tiny ARI loss on that correlated-ellipse fixture (0.499714 to 0.499700), despite improved BIC. Using two initializations recovered the intended groups on this development sample. The final comparison above was rerun for all eight cases; this tuning history is further reason to keep the method opt-in and use independent confirmation.

Nested circles and moons remain weaknesses. No general nonlinear-clustering claim is made. No network datasets, hidden contest data, or official scores were used.

Machine-readable final development results: `experiments/clustering-development42.json`.

## Tests

`tests/test_clustering_candidates.py`: eight tests covering deterministic ordered outputs, unmodified inputs, categorical route, constant-row k occupancy, JSON metadata, invalid group counts, low-budget covariance skip, and independence from supplied reference labels.

Command: `.venv312/Scripts/python.exe -m pytest tests/test_clustering_candidates.py -q --basetemp experiments/clustering-final-test42`.

Result: **8 passed**, one expected convergence warning for the constant-table case.
