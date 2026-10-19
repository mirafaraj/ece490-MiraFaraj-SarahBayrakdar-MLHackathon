# Independent supervised candidate cross-review

Scope: read-only review of `src/unknown_dataset/supervised.py`, its routing from `models.py`/`runner.py`, and `tests/test_supervised_candidates.py`. Production code, tests, scripts, benchmark settings, and confirmation outputs were not changed or inspected beyond source. No tests or timing experiments were launched during the frozen benchmark, to avoid competing with its CPU measurements.

## Finding: P2 — Low-budget route increases baseline tree workload

`supervised.py` `_fit`, line 47, always builds 96 Extra Trees for the final refit. `fit_predict_supervised`, lines 95–96, recognizes `time_budget < 12` and skips selection, but then calls that same 96-tree final refit. In contrast, `models.py` baseline uses 32 trees when `time_budget < 15`.

Consequently the auto route can triple tree workload even when it explicitly skips selection for lack of time. The hard worker deadline prevents an unbounded run, but may discard the fit and replace useful predictions with emergency majority/mean predictions. This is a concrete source-path difference, not a measured timeout regression; the frozen run was not disturbed to reproduce timing.

Recommended follow-up after the benchmark: pass an explicit bounded final tree count derived from available budget, or delegate skipped selection directly to the baseline implementation. Add a test verifying the actual estimator tree count on the low-budget route; current tests check only the skip reason, output validity, and empty score mapping.

## Leakage and model-selection checks

- No evaluation labels are consumed by production selection. Label encoding is learned from the supplied training labels only. Observing the full training class vocabulary before a training holdout is not evaluation-label leakage.
- Each candidate constructs a new cleaner and preprocessor, fitting both solely on the fitting subset. Validation features are transform-only. Full-data refitting happens only after model selection. Test rows are transform-only.
- Selection metric is explicitly balanced accuracy for classification and negative RMSE for regression. These remain provisional until official metrics are supplied. The code makes no assumption of official contest points.
- At most 4,000 rows enter selection, with one 25% holdout; no cross-validation search is hidden in the implementation. Classification stratifies both subsampling and holdout, and catches the small-sampled-class split failure.
- The initial selected model is Extra Trees. Failed candidate fits do not overwrite a working winner. If the chosen non-tree candidate cannot refit under its dense cap, the code retries Extra Trees and records the fallback reason.
- Validation Extra Trees uses 40 trees and final refit uses 96; histogram uses 60 versus 100 iterations. Candidate ranking therefore measures approximations rather than exactly the final estimators. This is an understandable runtime tradeoff, but validation rankings are not guarantees of final quality.

## Runtime and memory limitations

- Per-candidate clock checks reserve time between fits; an individual fit is interruptible only by the outer worker deadline. Final full-data refitting can be much larger than the 4,000-row selection problem. This is documented, not silently assumed safe.
- `_DENSE_CELLS` guards explicit sparse-to-dense conversion and bounds histogram prediction batches. It is **not a total memory cap**: the cleaner/preprocessor can already allocate a dense numeric matrix before its shape is checked, while tree/linear models may use large sparse matrices. The input skip threshold still allows 100,000 rows by 1,000 features (100 million numeric cells). Maximum-size and real memory measurements remain required.
- Histograms are selected on the training holdout without consulting evaluation/test sizes. Dense test prediction is batched correctly according to transformed feature width.
- Logistic convergence warnings are suppressed and no `n_iter_`/convergence metadata is recorded. Finite predictions can still be returned at the iteration cap. This limits diagnosis but is not itself proof of incorrect output.

## Test coverage assessment

Current candidate tests meaningfully exercise a linear signal, training-split preprocessing isolation, fractional class preservation on skipped selection, mixed missing categories and unseen test categories, dense candidate rejection, and constant/single-class data.

Additional useful checks after the freeze: low-budget tree-count behavior, final-refit dense-cap fallback, invalid/rare-class sampled split handling, and large test prediction batching. No high-severity correctness or leakage defect was identified in the reviewed source.
