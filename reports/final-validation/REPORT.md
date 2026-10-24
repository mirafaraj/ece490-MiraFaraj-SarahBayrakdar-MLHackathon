# Final validation record

Model commit tested: `380d56d4a7c2ec1e47bea9008fed442aa39c8054`.

## Execution checks

- 72 fresh, serial runs of the standard CLI using default submission strategy.
- 72 valid prediction outputs and independently checked full-training profiles.
- Zero crashes, external timeouts, invalid outputs or emergency predictions.
- Two runs recovered validated learned selection-model predictions (Bike and Metro).
- Maximum complete-process wall time: 47.55 seconds; all runs below the announced 60-second limit.
- All 63 available historical prediction comparisons matched.
- Main source hashes were unchanged throughout the campaign.
- The preceding clean-commit suite passed 355 tests; independent focused recovery testing passed 46 tests (an overlapping subset, not 46 additional unique suite tests). Those suites were not rerun as part of the 72-case campaign.

This record adds to the earlier 120-configuration compatibility campaign; counts from different campaigns are not added as unique datasets. The local Windows/Python 3.12 environment used the pinned requirements and one numerical/estimator CPU thread.

## Classification results

Values are percentages. Splits are averaged within each dataset before equal-dataset averaging.

| Dataset | Accuracy | Balanced accuracy | Macro F1 | Minority precision | Minority recall |
|---|---:|---:|---:|---:|---:|
| ai4i | 96.88 | 88.74 | 80.96 | 52.71 | 80.00 |
| bank | 82.98 | 74.42 | 68.19 | 36.76 | 63.24 |
| blood_transfusion | 66.49 | 68.95 | 62.77 | 39.15 | 73.65 |
| breast_cancer | 97.44 | 97.45 | 97.27 | 95.77 | 97.48 |
| digits | 96.74 | 96.71 | 96.71 | 93.43 | 88.37 |
| dry_bean | 91.77 | 93.45 | 93.32 | 100.00 | 100.00 |
| iris | 92.11 | 92.24 | 92.20 | 91.67 | 84.62 |
| mammography | 92.67 | 86.74 | 67.23 | 27.52 | 80.51 |
| telco | 75.87 | 77.21 | 72.84 | 52.97 | 80.09 |
| uci-seismic | 75.00 | 71.31 | 55.54 | 16.23 | 67.05 |
| uci-spambase | 95.79 | 95.64 | 95.59 | 94.41 | 94.93 |
| wine | 98.52 | 98.77 | 98.49 | 95.24 | 100.00 |
| Equal-dataset mean | 88.52 | 86.80 | 81.76 | 66.32 | 84.16 |

## Regression results

| Dataset | RMSE | MAE | R2 | RMSE reduction versus training mean |
|---|---:|---:|---:|---:|
| california_housing | 0.4849 | 0.3327 | 0.8241 | 58.08% |
| diabetes | 54.5265 | 43.5017 | 0.4864 | 28.54% |
| house_prices | 30288.3319 | 17860.9619 | 0.8630 | 63.55% |
| uci-bike-hour | 98.8703 | 68.8190 | 0.7980 | 58.20% |
| uci-metro-traffic | 1909.9455 | 1640.9387 | 0.0731 | 3.76% |

RMSE and MAE use each target's units; they are not averaged across datasets. Bike and Metro recovered models fitted on 3,000 rows, with profiles covering all training rows.

## Clustering results

| Dataset | ARI | AMI |
|---|---:|---:|
| digits | 0.5440 | 0.6754 |
| iris | 0.9039 | 0.8984 |
| wine | 0.8862 | 0.8649 |

Both Mall inputs produced the requested five groups. No reference group labels were available for Mall, so no ARI/AMI or accuracy is claimed.

## Panel and interpretation

The panel includes 37 classification, 15 regression and 20 clustering calls. It covers existing real-data splits, large/wide tables, mixed-feature probes, missingness, high-cardinality categoricals, duplicate/constant features, lexical labels, multiclass inputs, and reference-labelled clustering edge cases. Per-case results, timings and recovery metadata are in [per-case.csv](per-case.csv). Raw datasets, private references and full predictions remain local.

Accuracy, balanced accuracy, F1, precision, recall, R2, ARI and AMI are higher-is-better; RMSE/MAE are lower-is-better. ARI/AMI near zero mean chance-level reference agreement, not zero percent correct. R2 may be negative. No diagnostic metric is asserted to be the official grading formula.

The 25-class synthetic case achieved 10.00% accuracy. Its existing generator assigns equal-prior classes 0.2-spaced means in one unit-variance Gaussian feature, with independent noise features. The population Bayes accuracy under that construction is approximately 11.65%, compared with 4% uniform guessing. This is an analytical interpretation, not a separately fitted or tuned model.

All cases were fixed before fitting, with frozen input/reference hashes and source commit. No dataset substitution, quality tuning, model change, weight/threshold change or experimental strategy promotion occurred. Existing inspected datasets and repeated splits are validation evidence, not fresh proof of unseen-dataset performance. The current approach, interface assumptions and interpretation are documented in the [README](../../README.md) and [model card](../../MODEL_CARD.md).
