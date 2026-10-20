# Real-data check: how to read and reproduce it

Everything created for this request lives in this folder. Existing project code,
configuration, evaluation framework and earlier reports are left untouched.

- `FACTS.md`: commands, documented contract versus actual behavior, source line numbers.
- `STUDENT_SUMMARY.md`: final plain-English findings and recommendations.
- `REPORT.md` / `summary.json`: all means, paired gains, worst dataset means,
  worst single seeds, every individual metric loss, and reliability alerts.
- `classification-table`, `regression-table`, `clustering-table` (`.md` / `.csv`):
  one row per dataset/seed, with each strategy's metrics, time and selected model.
- `stress-table.md` / `.csv`: task, shape/edge case, time and pass/fail per strategy.
- `manifest.json`: authoritative frozen inventory, dataset sources, versions,
  input hashes, code hashes and preservation snapshot.
- `results.jsonl`: one durable record per actual CLI invocation.
- `inputs/`: metadata.json and training/test CSVs supplied to the program.
- `references/`: scoring labels, deliberately outside the program's input folders.
- `runs/`: exact commands, stdout/stderr, and the program's actual output files.
- `download-cache/`: public datasets fetched through scikit-learn.

`check.py` is the single runner. From the repository root:

```powershell
# Resume only missing runs; completed identities are not repeated.
.\.venv312\Scripts\python.exe -B experiments/real-data-check/check.py --run

# Rebuild tables from saved results without fitting anything.
.\.venv312\Scripts\python.exe -B experiments/real-data-check/check.py --report
```

For a fresh folder with no manifest, use `--prepare --allow-download` first.
The preparation-attempt manifests and unused fallback inputs preserve early
acquisition diagnostics. The final manifest/table inventory uses the actual
mammography, blood-transfusion, California housing and House Prices downloads;
the breast-cancer fallback variants are not included in the benchmark.

Real classification uses stratified 75/25 splits; regression uses 75/25 splits.
Clustering uses all unlabeled feature rows, with k supplied and labels held only
by the scorer. Seeds are42/314/2027. Every strategy uses the actual CLI, its normal
configuration, an internal50s budget and an external60s guard. No experimental
third-batch control or lazy-import flag is enabled.

Stress pass means valid contract output **and** completion below60s; it does not
mean a useful predictor. Emergency outputs are scored and marked. An invalid
call can have unavailable selection/fallback information because the CLI removes
failed outputs; a missing field must not be interpreted as proof that no fallback
was attempted. Successful emergency calls show the final fallback model, not the
unknown estimator that was running before interruption.

These are comparisons on this Windows CPU environment, not official contest
scores or a clean assessment-environment certification. Three seeds of the same
dataset are not three independent datasets. No parameters were tuned on test
labels. No commits or pushes are part of this request.
