# Independent review: evaluation and tests

Reviewed `scripts/evaluate.py`, all four test modules, CI, pinned requirements, and saved benchmark. Review date: 2026-10-19. No production files changed during review.

## Confirmed findings

- **P2: Clustering checker is tied to three groups.** `evaluate_case` requires `len(unique(pred)) == 3`, ignoring a dataset's requested cluster count. Valid two/four-group cases would be rejected. Store `n_clusters` in case metadata and validate against it; also validate integer labels in the allowed range.
- **P2: Classification metric computation rejects valid fractional labels.** Production tests explicitly support classes `0.25` and `1.75`, but `accuracy_score([0.25, 1.75], [0.25, 1.75])` raises `ValueError: continuous is not supported`. Encode truth/predictions to stable discrete identifiers before classification metrics. Preserve raw-label validation separately.
- **P2: Profiling validity only means nonempty JSON.** The evaluator accepts any nonempty dictionary as `profile.json`; incorrect counts, feature types, missingness, constants, duplicate columns, and target summaries can pass. Independently derive expected raw-data values and compare required provisional fields. Official scoring semantics remain unknown.

## Coverage and interpretation limits

- All 34 current tests pass with `.venv312/Scripts/python.exe -m pytest -q --basetemp experiments/review-evaluation-tmp` (13.22s). The expected identical-points K-means convergence warning is present.
- A plain pytest run initially had 20 temp-fixture setup errors from restricted sandbox temp permissions, with 14 tests passing. Explicit writable workspace basetemp resolves this environment issue; it is not a production defect.
- Existing fallback tests call the fallback directly. They do not force a real long-running worker, prove it is terminated, verify no orphan remains, or prove final artifacts arrive within the hard deadline.
- There are no evaluator unit tests, cross-seed determinism tests, clean-install evidence, or maximum-size/memory checks.
- The three benchmark datasets use one seed and one family per task. The clustering task has extremely separated centers plus a category directly corresponding to truth. Its ARI=1.0 is plausible but weak evidence of general clustering performance.
- Supervised synthetic targets are generated before corruption and reference labels remain in the evaluation process; no evaluation-label leakage into the production CLI was found.
- Current reports record dependencies, generator version, and seed, but commit is null. Add source fingerprint, dirty status where Git exists, candidate configuration, case name/source, split seed, and a data fingerprint for comparisons before Git initialization and with uncommitted changes.
- CI runs Ubuntu Python 3.12; no successful remote CI or clean pinned installation has been observed. Windows-only local success cannot establish assessment-environment compatibility.
- Accuracy/balanced accuracy, RMSE/R2, and ARI are development metrics; no official aggregate points can be claimed until scoring rules arrive.

## Proposed broadened suite interface

Implement `scripts/benchmark_cases.py` with named case factories and a common case object containing name, task, input directory, evaluation reference, training target, source, seed, and requested cluster count. Keep references only in the harness. `make_cases(root, seed, suite="smoke", names=None)` should offer smoke and broad suites. Use identical generated tables and fixed splits for every candidate.

Broader offline families: breast cancer, wine/iris, digits (classification); diabetes, linear and nonlinear synthetic regression; anisotropic/unequal-density blobs, moons, numeric irrelevant-feature clustering; mixed categorical/missing/high-cardinality stress cases. Use only locally bundled sklearn loaders and generators, without downloads. Preserve a second seed as confirmation rather than using every seed to design model policies.

Extend evaluator with `--suite`, `--seeds`, `--cases`, and explicit `--strategy`; unique per-case output directories; row-level failures; per-task summaries; robust classification encoding; cluster-count checks; and independent basic profile checks. Keep full CLI wall-clock measurements distinct from faster direct-model exploration timings.

## Implemented follow-up

Added twelve named offline families in `scripts/datasets.py` (four per task), multi-seed and baseline/auto candidate controls in `scripts/evaluate.py`, data and source SHA-256 fingerprints, source-change rejection, generalized cluster validation, lexical classification prediction parsing, and independent raw-data profile checks. Classification target profiling follows opaque CSV-token semantics, preserving classes such as `01` and `1`. No reference labels are passed to the production program. Five harness tests pass, including an actual CLI evaluation preserving those distinct labels. The full 48-run benchmark is delegated to the orchestrator after model candidates and source files are frozen.
