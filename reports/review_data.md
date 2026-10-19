# Independent data, profiling, and preprocessing review

Reviewed `contract.py`, `profiling.py`, `preprocessing.py`, related tests, and README on 2026-10-19.
The official contest contract is unavailable; findings apply to the provisional documented contract.

## Findings

1. **P1 — Classification labels silently collapse during CSV inference.** `contract.py:30,71` reads the target with pandas dtype inference. A CSV target containing `01` and `1` loads as `[1,1,1,1]`, losing an entire class before modelling. Preserve lexical classification labels by default and allow explicit numeric target typing when the actual schema requires it. Add loader coverage for leading zeros and fractional labels.
2. **P2 — Numeric missingness signal disappears after retention.** `preprocessing.py:89` median-imputes without an indicator. Input `[7, missing, 7, missing]` is retained by the cleaner but becomes `[7,7,7,7]`. This contradicts the intended missingness preservation and prevents learning a perfectly predictive missingness signal. Add train-fitted missingness indicators.
3. **P2 — Categorical sentinel collides with legitimate data.** `preprocessing.py:95` uses the literal `__MISSING__` for imputation. Input `["__MISSING__", missing, "other"]` encodes the first two rows identically (`[1,0]`). Use disjoint representations for real categories and missing values.
4. **P2 — Inferred categorical values can change across CSV parses.** `contract.py:30` reads test dtypes using only explicitly declared types. When feature types are inferred from training, a string category such as `1` may be parsed as numeric `1.0` in a test column containing missing values and subsequently normalized to a different string. Parse test categoricals using the final feature types learned from training.

## Validation evidence

- Existing contract/profile suite: **21 passed** using `.venv312/Scripts/python.exe -m pytest tests/test_contract.py tests/test_profiling.py -q --basetemp .pytest_tmp_review_data`.
- The initial sandbox run failed to create pytest temporary directories (18 setup errors, 3 passed). The approved escalated rerun passed; these were environment errors, not application failures.
- Minimal direct reproductions confirmed findings 1–3 with the exact values above.
- No train/test fitting leakage found in the inspected preprocessing path: cleaner and preprocessor are fitted only on training features.
- Profiling intentionally counts constants using distinct nonmissing values and requires matching dtype for duplicate columns; README documents both choices. Their official scoring compatibility remains unverified.

## Follow-up

Production fixes and dedicated regression tests were authorized after the independent review. Final regression results are recorded below once the fixes are verified.

## Fixes and regression results

- Classification targets are read as lexical strings through the file adapter; direct model API numeric labels remain supported. Inferred categorical test columns now use the final training feature types.
- Numeric imputation adds training-fitted missingness indicators. Real categorical values are prefixed into a namespace disjoint from the missing-value sentinel.
- At the orchestrator's request, output publication now stages every file before replacement, commits predictions last, and clears the three known output artifacts after partial publication failure. Staging failures leave the previous complete result intact. This is consistency protection, not a filesystem transaction; process crashes or unremovable files remain OS-level limitations.
- `tests/test_review_data.py` adds seven regression cases, including two injected output I/O failures.
- Verification: **39 passed**, one expected coincident-point clustering warning, using `.venv312/Scripts/python.exe -m pytest tests/test_review_data.py tests/test_contract.py tests/test_profiling.py tests/test_models.py -q --basetemp .pytest_tmp_review_data_fixed` (5.05 seconds test time).
- The official schema, lexical-vs-numeric label semantics, profiling definitions, and maximum-size behavior still require contest materials.
