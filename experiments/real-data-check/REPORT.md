# Real-data comparison and stress results

Completed **147/147** actual CLI runs.

All figures are measurements on this machine, not official hackathon points. Classification uses stratified75/25 splits; regression uses75/25 splits. Clustering receives all feature rows and k, never labels.

Seeds: (42, 314, 2027). Internal budget50s; external guard60s. No lazy imports or experimental controls enabled. Current working-tree code is frozen by hashes, including existing uncommitted third-batch controls in their disabled defaults.

## Facts and complete tables

See [FACTS.md](FACTS.md), [classification](classification-table.md), [regression](regression-table.md), [clustering](clustering-table.md), and [stress](stress-table.md). CSV counterparts are suitable for Excel/Colab.

Minority recall means recall averaged over the least frequent training class(es), ties averaged. It is descriptive on balanced datasets.

## Task means and worst losses

Classification gains are percentage-point differences; clustering gains are absolute ARI/AMI differences. Regression RMSE/MAE gains are within-dataset relative reductions: positive is better. Raw mean regression errors mix different target units and should not determine the winner.

| Task | Strategy | Metric | Mean metric | Mean gain vs baseline | Worst dataset mean gain | Worst single dataset/seed gain |
|---|---|---|---:|---:|---|---|
| classification | baseline | accuracy | 0.92228 | +0.000 pp | breast_cancer +0.000 pp | breast_cancer/42: +0.000 pp |
| classification | baseline | balanced_accuracy | 0.86238 | +0.000 pp | breast_cancer +0.000 pp | breast_cancer/42: +0.000 pp |
| classification | baseline | macro_f1 | 0.87229 | +0.000 pp | breast_cancer +0.000 pp | breast_cancer/42: +0.000 pp |
| classification | baseline | minority_recall | 0.76516 | +0.000 pp | breast_cancer +0.000 pp | breast_cancer/42: +0.000 pp |
| classification | auto | accuracy | 0.90659 | -1.569 pp | mammography -5.889 pp | mammography/314: -8.512 pp |
| classification | auto | balanced_accuracy | 0.90140 | +3.902 pp | digits -1.262 pp | digits/2027: -1.577 pp |
| classification | auto | macro_f1 | 0.85778 | -1.451 pp | mammography -12.807 pp | mammography/42: -19.829 pp |
| classification | auto | minority_recall | 0.88294 | +11.779 pp | digits -3.101 pp | digits/2027: -6.977 pp |
| classification | routed | accuracy | 0.90659 | -1.569 pp | mammography -5.889 pp | mammography/314: -8.512 pp |
| classification | routed | balanced_accuracy | 0.90140 | +3.902 pp | digits -1.262 pp | digits/2027: -1.577 pp |
| classification | routed | macro_f1 | 0.85778 | -1.451 pp | mammography -12.807 pp | mammography/42: -19.829 pp |
| classification | routed | minority_recall | 0.88294 | +11.779 pp | digits -3.101 pp | digits/2027: -6.977 pp |
| regression | baseline | rmse | 10281.55976 | +0.000% | diabetes +0.000% | diabetes/42: +0.000% |
| regression | baseline | mae | 6265.51127 | +0.000% | diabetes +0.000% | diabetes/42: +0.000% |
| regression | baseline | r2 | 0.71082 | +0.000 | diabetes +0.000 | diabetes/42: +0.000 |
| regression | auto | rmse | 10114.44777 | +2.315% | house_prices +1.866% | diabetes/42: -3.073% |
| regression | auto | mae | 5968.26547 | +2.468% | california_housing -1.380% | california_housing/2027: -2.259% |
| regression | auto | r2 | 0.72453 | +0.014 | house_prices +0.004 | diabetes/42: -0.030 |
| regression | routed | rmse | 10114.44777 | +2.315% | house_prices +1.866% | diabetes/42: -3.073% |
| regression | routed | mae | 5968.26547 | +2.468% | california_housing -1.380% | california_housing/2027: -2.259% |
| regression | routed | r2 | 0.72453 | +0.014 | house_prices +0.004 | diabetes/42: -0.030 |
| clustering | baseline | ari | 0.68233 | +0.000 | iris +0.000 | iris/42: +0.000 |
| clustering | baseline | ami | 0.73108 | +0.000 | iris +0.000 | iris/42: +0.000 |
| clustering | auto | ari | 0.77802 | +0.096 | wine +0.000 | wine/42: +0.000 |
| clustering | auto | ami | 0.81293 | +0.082 | wine +0.000 | wine/42: +0.000 |
| clustering | routed | ari | 0.77802 | +0.096 | wine +0.000 | wine/42: +0.000 |
| clustering | routed | ami | 0.81293 | +0.082 | wine +0.000 | wine/42: +0.000 |

## Every measured loss versus baseline

| Task | Dataset/seed | Strategy | Metric | Baseline | Candidate | Gain (negative = loss) |
|---|---|---|---|---:|---:|---:|
| classification | blood_transfusion/314 | auto | accuracy | 0.67914 | 0.64171 | -0.03743 |
| classification | blood_transfusion/314 | routed | accuracy | 0.67914 | 0.64171 | -0.03743 |
| classification | blood_transfusion/2027 | auto | accuracy | 0.70053 | 0.65775 | -0.04278 |
| classification | blood_transfusion/2027 | routed | accuracy | 0.70053 | 0.65775 | -0.04278 |
| classification | digits/42 | auto | accuracy | 0.98222 | 0.96667 | -0.01556 |
| classification | digits/42 | auto | balanced_accuracy | 0.98186 | 0.96634 | -0.01551 |
| classification | digits/42 | auto | macro_f1 | 0.98198 | 0.96628 | -0.01570 |
| classification | digits/42 | auto | minority_recall | 0.90698 | 0.88372 | -0.02326 |
| classification | digits/42 | routed | accuracy | 0.98222 | 0.96667 | -0.01556 |
| classification | digits/42 | routed | balanced_accuracy | 0.98186 | 0.96634 | -0.01551 |
| classification | digits/42 | routed | macro_f1 | 0.98198 | 0.96628 | -0.01570 |
| classification | digits/42 | routed | minority_recall | 0.90698 | 0.88372 | -0.02326 |
| classification | digits/314 | auto | accuracy | 0.98444 | 0.97778 | -0.00667 |
| classification | digits/314 | auto | balanced_accuracy | 0.98424 | 0.97767 | -0.00657 |
| classification | digits/314 | auto | macro_f1 | 0.98424 | 0.97760 | -0.00664 |
| classification | digits/314 | routed | accuracy | 0.98444 | 0.97778 | -0.00667 |
| classification | digits/314 | routed | balanced_accuracy | 0.98424 | 0.97767 | -0.00657 |
| classification | digits/314 | routed | macro_f1 | 0.98424 | 0.97760 | -0.00664 |
| classification | digits/2027 | auto | accuracy | 0.97333 | 0.95778 | -0.01556 |
| classification | digits/2027 | auto | balanced_accuracy | 0.97291 | 0.95714 | -0.01577 |
| classification | digits/2027 | auto | macro_f1 | 0.97285 | 0.95756 | -0.01529 |
| classification | digits/2027 | auto | minority_recall | 0.88372 | 0.81395 | -0.06977 |
| classification | digits/2027 | routed | accuracy | 0.97333 | 0.95778 | -0.01556 |
| classification | digits/2027 | routed | balanced_accuracy | 0.97291 | 0.95714 | -0.01577 |
| classification | digits/2027 | routed | macro_f1 | 0.97285 | 0.95756 | -0.01529 |
| classification | digits/2027 | routed | minority_recall | 0.88372 | 0.81395 | -0.06977 |
| classification | mammography/42 | auto | accuracy | 0.98677 | 0.90451 | -0.08226 |
| classification | mammography/42 | auto | macro_f1 | 0.82044 | 0.62215 | -0.19829 |
| classification | mammography/42 | routed | accuracy | 0.98677 | 0.90451 | -0.08226 |
| classification | mammography/42 | routed | macro_f1 | 0.82044 | 0.62215 | -0.19829 |
| classification | mammography/314 | auto | accuracy | 0.98534 | 0.90021 | -0.08512 |
| classification | mammography/314 | auto | macro_f1 | 0.79330 | 0.61273 | -0.18056 |
| classification | mammography/314 | routed | accuracy | 0.98534 | 0.90021 | -0.08512 |
| classification | mammography/314 | routed | macro_f1 | 0.79330 | 0.61273 | -0.18056 |
| classification | mammography/2027 | auto | accuracy | 0.98462 | 0.97532 | -0.00930 |
| classification | mammography/2027 | auto | macro_f1 | 0.78735 | 0.78199 | -0.00536 |
| classification | mammography/2027 | routed | accuracy | 0.98462 | 0.97532 | -0.00930 |
| classification | mammography/2027 | routed | macro_f1 | 0.78735 | 0.78199 | -0.00536 |
| regression | california_housing/42 | auto | mae | 0.33111 | 0.33195 | -0.00254 |
| regression | california_housing/42 | routed | mae | 0.33111 | 0.33195 | -0.00254 |
| regression | california_housing/314 | auto | mae | 0.33082 | 0.33621 | -0.01628 |
| regression | california_housing/314 | routed | mae | 0.33082 | 0.33621 | -0.01628 |
| regression | california_housing/2027 | auto | mae | 0.32275 | 0.33004 | -0.02259 |
| regression | california_housing/2027 | routed | mae | 0.32275 | 0.33004 | -0.02259 |
| regression | diabetes/42 | auto | r2 | 0.51610 | 0.48590 | -0.03020 |
| regression | diabetes/42 | auto | rmse | 51.72851 | 53.31824 | -0.03073 |
| regression | diabetes/42 | routed | r2 | 0.51610 | 0.48590 | -0.03020 |
| regression | diabetes/42 | routed | rmse | 51.72851 | 53.31824 | -0.03073 |

## Crashes, invalid results, and calls over45seconds

All alerts are listed below; valid deadline fallback is not labeled a crash.

- stress-classification-shape_100000x50-7001, baseline: 50.90s; valid=False; timeout=False; fallback=None; error=ERROR: Output serialization exceeded the execution budget
.

- stress-classification-shape_100000x50-7001, routed: 54.01s; valid=False; timeout=False; fallback=None; error=ERROR: Output serialization exceeded the execution budget
.

- stress-regression-shape_100000x50-7001, baseline: 47.33s; valid=True; timeout=False; fallback=emergency; error=none.
- stress-regression-shape_100000x50-7001, routed: 47.58s; valid=True; timeout=False; fallback=emergency; error=none.
- stress-regression-shape_100000x50-7001, auto: 47.41s; valid=True; timeout=False; fallback=emergency; error=none.
- stress-classification-shape_5000x500-7001, auto: 47.56s; valid=True; timeout=False; fallback=emergency; error=none.
- stress-regression-shape_5000x500-7001, auto: 47.38s; valid=True; timeout=False; fallback=emergency; error=none.
- stress-regression-shape_5000x500-7001, baseline: 47.37s; valid=True; timeout=False; fallback=emergency; error=none.
- stress-regression-shape_5000x500-7001, routed: 47.52s; valid=True; timeout=False; fallback=emergency; error=none.

## Maximum full-process wall time

All runs: {'baseline': 50.899151799996616, 'auto': 47.56156850000116, 'routed': 54.011448199998995}. Stress only: {'baseline': 50.899151799996616, 'auto': 47.56156850000116, 'routed': 54.011448199998995}.

## Dataset provenance and unavailable work

- {"dataset": "breast_cancer", "task": "classification", "status": "available", "rows": 569, "features": 30, "categories": 0, "source_class_counts": {"0": 212, "1": 357}, "source": "sklearn bundled real dataset"}
- {"dataset": "wine", "task": "classification", "status": "available", "rows": 178, "features": 13, "categories": 0, "source_class_counts": {"0": 59, "1": 71, "2": 48}, "source": "sklearn bundled real dataset"}
- {"dataset": "digits", "task": "classification", "status": "available", "rows": 1797, "features": 64, "categories": 0, "source_class_counts": {"0": 178, "1": 182, "2": 177, "3": 183, "4": 181, "5": 182, "6": 181, "7": 179, "8": 174, "9": 180}, "source": "sklearn bundled real dataset"}
- {"dataset": "iris", "task": "classification", "status": "available", "rows": 150, "features": 4, "categories": 0, "source_class_counts": {"0": 50, "1": 50, "2": 50}, "source": "sklearn bundled real dataset"}
- {"dataset": "mammography", "task": "classification", "status": "available", "rows": 11183, "features": 6, "categories": 0, "source_class_counts": {"-1": 10923, "1": 260}, "source": "OpenML real dataset", "data_id": 310, "reported_name": "mammography", "url": "https://www.openml.org/d/310"}
- {"dataset": "blood_transfusion", "task": "classification", "status": "available", "rows": 748, "features": 4, "categories": 0, "source_class_counts": {"2": 178, "1": 570}, "source": "OpenML real dataset", "data_id": 1464, "reported_name": "blood-transfusion-service-center", "url": "https://www.openml.org/d/1464"}
- {"dataset": "diabetes", "task": "regression", "status": "available", "rows": 442, "features": 10, "categories": 0, "source_class_counts": null, "source": "sklearn bundled real dataset"}
- {"dataset": "california_housing", "task": "regression", "status": "available", "rows": 20640, "features": 8, "categories": 0, "source_class_counts": null, "source": "sklearn fetched real dataset"}
- {"dataset": "house_prices", "task": "regression", "status": "available", "rows": 1460, "features": 80, "categories": 43, "source_class_counts": null, "source": "OpenML real categorical regression", "data_id": 42165, "url": "https://www.openml.org/d/42165", "reported_name": "house_prices"}

## Preservation

Outside-folder files preserved: **True**. Production source/config hashes match: **True**.
No models, defaults, configurations, existing evaluation reports, commits or remote branches were changed. Inputs/references/logs/downloads and this harness all stay under experiments/real-data-check/.

## Interpretation

See STUDENT_SUMMARY.md for the final plain-English evidence and recommendations. Until all planned runs complete, these tables are partial and cannot establish a winner.
Three seeds are repeated splits of the same dataset, not independent new datasets. No model was tuned against these held-out results. Official scoring, clean assessment environment and memory compliance remain unverified.
