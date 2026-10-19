# Stage 1 independent runtime review

Reviewed `main.py`, `runner.py`, `fallback.py`, `models.py`, output publication, and associated tests. No production changes made by this reviewer. Findings refer to the original MVP before integration fixes.

## Findings

### P2 — Failed reruns retain successful-looking artifacts from a previous run

Location: `src/unknown_dataset/runner.py`, original lines 121–123 (exception handler), and output setup before worker launch.

Reproduction: seed an existing output directory with `predictions.csv`, `profile.json`, and `run.json` containing `PRIOR_RUN_SENTINEL`; call the runner against metadata with task `unsupported`. The runner returns 1, but all three old files remain unchanged. A consumer that reads artifacts without checking the exit status can score a previous dataset's predictions. Existing tests only check failure against a fresh output directory.

Recommended fix: explicitly invalidate only the three program-owned artifacts at the start of an invocation (after checking input/output path safety), and invalidate any partially published outputs on failure. Do not recursively clear the user's output directory. Add a failed-rerun test with existing artifacts and a sentinel unrelated user file.

### P2 — Output write failures can mix new predictions with old metadata

Location: `src/unknown_dataset/contract.py`, `write_outputs`, sequential CSV and JSON writes; `runner.py` original lines 115–118 publish before checking elapsed budget.

Reproduction: seed prior artifacts, patch `Path.write_text` to raise `OSError` for `profile.json`, and call `write_outputs` with predictions `[1, 2]`. The observed directory has new `predictions.csv` alongside old `profile.json` and `run.json`. A serialization-time deadline failure also occurs after outputs have been published.

Recommended fix: stage every output before publication; on any write/publication/deadline error remove program-owned final artifacts. Single-file replacement helps prevent truncation, but replacing three files is not itself an atomic group transaction. Add an injected mid-publication failure test.

## Verified behavior

- `.venv312/Scripts/python.exe -m pytest tests/test_cli.py tests/test_models.py -q --basetemp experiments/runtime-review-pytest`: **13 passed**, one expected K-means convergence warning on identical points.
- First sandbox attempt: 11 model tests passed, 2 CLI tests errored during pytest temporary-directory setup. Escalated rerun above passed; these were environment errors, not application failures.
- Actual CLI on a four-row classification fixture with budgets 2 and 3 seconds terminated the unfinished worker and emitted valid majority-class fallback outputs. Observed external wall times 2.186 and 3.096 seconds; internally reported elapsed times 1.766 and 2.578 seconds. No hanging worker was observed.
- `--time-budget nan` and `--time-budget inf` both rejected with exit code 2 and no run artifacts.
- Unexpected model exceptions deliberately fail nonzero; the orchestrator confirmed this is intended and documented. It is not a defect requiring silent fallback.
- Supervised target encoding is fitted on training labels; predictions are decoded to original labels and checked against the training label set. Training-only preprocessing avoids test-feature leakage in the reviewed path.

## Remaining verification limitations

- The budget clock starts inside `runner.main`, after interpreter and entry-point startup. Full wall time can exceed the requested internal budget by startup overhead. The default 50-second budget leaves headroom beneath the 60-second contest limit, but the final assessment environment must measure external wall time.
- Input loading, profiling, process startup serialization, fallback, and output are not isolated behind an outer hard deadline. Worker termination is covered; a hard full-process 60-second watchdog still belongs to the assessment/evaluation harness. Maximum-size tests await official size limits.
- No claim of Windows memory-cap compliance or Linux clean-install reproducibility is made by this review.

Reproduction fixture artifacts were written under `experiments/runtime-review-repros/`; they are development-only and do not modify the baseline report.
