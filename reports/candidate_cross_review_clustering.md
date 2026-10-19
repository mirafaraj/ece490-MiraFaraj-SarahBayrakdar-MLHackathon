# Independent cross-review: clustering candidate

Reviewed `src/unknown_dataset/clustering.py` and `tests/test_clustering_candidates.py` after candidate freeze. Production, tests, and scripts were not modified. No confirmation benchmark results were inspected.

## Conclusion

No release-blocking implementation defect found. Geometry selection should remain opt-in: its density criterion is not evidence of semantic clustering quality. Existing tests and an additional independent anisotropic smoke case passed.

## Correctness and leakage checks

- Neither `y_train` nor `X_test` is read. The test using `y_train=object()` confirms that reference-label contents do not affect outputs.
- Cleaner/preprocessor use the supplied training features only. Clustering has no held-out-label selection step.
- Prediction order follows the input matrix; sampling is used only for mixture fitting, followed by prediction on the original full matrix.
- Group counts validate integer type, exclude booleans, and require `1 <= k <= rows`. Missing groups are occupied by splitting a largest occupied group. Given this constraint, there is always a donor with at least two rows whenever an empty group exists.
- Required group occupancy is maintained even for identical rows. These repairs are arbitrary geometry-free assignments and may hurt quality, but metadata records them.

## Runtime and memory findings

- The covariance path permits at most 20,000 rows, 16 retained numeric features, and 12 groups. Missing indicators can double the transformed numeric dimension to 32; dense input is therefore still bounded to about 640,000 values. GaussianMixture fits no more than 2,000 rows, two initializations, and 80 iterations per covariance model.
- **Operational limitation:** time checks between estimators cannot interrupt the estimator already running. If optional GMM fitting consumes the outer worker deadline, the previously computed K-means result is lost and the runner may emit emergency fallback predictions. The external process deadline protects termination, but does not preserve useful completed work. Future mitigation: checkpoint the baseline result before optional experiments, or allocate optional work a separately enforced deadline. This is an improvement opportunity shared with other candidate selection paths, not a newly introduced uncontrolled-process defect.
- The initial K-means/MiniBatchKMeans work inherits the baseline's dataset-size/runtime risks and still needs published-limit stress tests.

## Selector risk and test gaps

- BIC compares spherical versus full-covariance mixtures on the same training sample. It does **not** compare semantic quality of full-covariance GMM against K-means, and lower BIC can recover density components that differ from assessment groups. The conservative margin does not resolve that mismatch. Keep the opt-in status and rely on untouched confirmation evidence for promotion.
- GMM is deliberately fitted to a maximum of 2,000 sampled rows and is not refitted to all rows. Rare groups may be missed; occupancy repair cannot reconstruct their geometry.
- Existing automated tests do not explicitly require the `GaussianMixtureFull` selection branch or simulate a failed/nonconvergent mixture fit. Add those focused regression cases after benchmark freeze if this strategy remains supported.

## Executed verification

- `.venv312/Scripts/python.exe -m pytest tests/test_clustering_candidates.py -q --basetemp .pytest_tmp_cross_clustering`: **8 passed**, one expected identical-point clustering warning; 2.49 seconds test time.
- Independent synthetic seed-42 anisotropic case, 300 rows/two numeric features/two groups: selected `GaussianMixtureFull`; spherical BIC 1534.36 versus full BIC 295.52; produced 300 assignments and exactly two groups; 0.157 seconds model time. This verifies the branch mechanically, not external clustering quality or worst-case speed.
