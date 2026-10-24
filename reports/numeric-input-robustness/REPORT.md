# Numeric input reliability and bounded diagnostics

Base/main: `9877532c8c44a420754c1df29ac87f69b89c09dd`; isolated branch `codex/numeric-input-robustness`. No model, selection objective, weighting, threshold, default, preprocessing implementation, dependency or execution-limit changes. Submission still delegates all tasks to existing auto. No commit/merge/push.

## Parsing policy (implementation assumption)

Only FEATURE columns explicitly declared numeric are converted with pandas.to_numeric(errors="coerce") in both train and test. Whitespace-only and other unparseable entries become missing; parsed numeric values are preserved. Infinity remains an error. Missing/invalid regression targets remain strict errors; classification targets remain opaque lexical class tokens with existing missing-target rejection. IDs and categorical values are untouched. Inferred feature-type behavior and CSV missing-token recognition are unchanged. No currency/unit/date guessing.

run.json gains numeric_feature_coercions, with train/test maps keyed only by declared numeric features. Each record has already_missing_count (missing after existing CSV parsing), coerced_whitespace_count and coerced_invalid_count (nonblank). The last two counts describe newly missing entries and do not include already-missing entries. Empty test maps are retained for clustering; inputs without declared types report an empty object. Counts contain no raw values/rows. Existing run/profile fields retain their meanings. Profiles describe the full converted training feature table, not a fit sample. Entirely missing features use existing preprocessing.

## Reliability and equivalence

- Raw unmodified synthetic Telco now succeeds in 11.30s: exactly 56 train and 30 test TotalCharges whitespace entries coerced. Predictions, selected LogisticRegression and full profile match the previously saved missing-token-normalized diagnostic. Original source hashes are unchanged. This peer fixture is synthetic, not the real Telco customer table.
- All 63 distinct saved valid input feature/target tables are identical to unchanged main, including dtypes and row order. Counts were independently reconstructed with per-token Python parsing, not the production conversion helper.
- Sixteen existing valid CLI configurations plus the newly accepted raw Telco call matched saved prediction hashes, model/candidate selections and profiles. Includes baseline/auto/routed on each task, default commands, all-missing/constant/duplicate inputs, real Telco, House Prices and clean Iris. This is a focused regression check, not a rerun of every historical campaign.
- All 17 calls valid learned output, zero emergency fallback; maximum 33.07s. Invocation budgets stayed 50s. Relevant probes:

| Existing probe | CLI strategy | Wall seconds | Outcome |
|---|---|---:|---|
| numeric 10,000x100 | auto | 31.16 | learned, same predictions/model/profile |
| mixed 45,000x50 | auto | 15.09 | learned, same predictions/model/profile |
| numeric 10,000x100 | default submission | 33.07 | learned, same predictions/model/profile |
| numeric 50,000x30 | default submission | 25.13 | learned, same predictions/model/profile |

- Focused loader/profile/CLI tests: 49 passed. Full suite: 338 passed. Both used a dedicated workspace pytest temp directory with its parent created beforehand. Tests cover nonblank invalid values, whitespace, pre-existing missingness, all-missing features, infinity rejection in train/test, strict regression targets, missing class labels, lexical spelling, source bytes and output counts.
- Independent profile reconstruction uses the unchanged historical reference. That strict checker initially rejected raw whitespace too; its reference was redirected locally to the preserved normalized table only after asserting the raw repaired and normalized full feature/target tables are identical. This was a local harness limitation; no evaluator/reference code changed and no CLI fit was repeated.

## Classification weighting diagnostic (reporting only)

Exactly four fixed full-training fits on the existing synthetic Telco split, seed42, unchanged existing _fit(final=True) estimators and preprocessing. Within each family only class_weight changed. Both pairs have24 encoded features; estimator parameters excluding class_weight match. Current auto is the raw CLI reference and selects balanced logistic; its predictions exactly match the separate balanced-logistic fit. Answers were read only for reporting, never for selection, threshold choice or parameters. These are inspected development results, not untouched confirmation. Supports: No1170, Yes330.

| Candidate | Accuracy % | BA % | Macro-F1 % | No P/R % | Yes P/R % | TN/FP/FN/TP | Seconds |
|---|---:|---:|---:|---|---|---|---:|
| Current auto (complete CLI) | 70.87 | 71.64 | 65.72 | 90.23/70.26 | 40.92/73.03 | 822/348/89/241 | 11.297 |
| linear-balanced (fit) | 70.87 | 71.64 | 65.72 | 90.23/70.26 | 40.92/73.03 | 822/348/89/241 | 0.373 |
| linear-unweighted (fit) | 82.73 | 69.24 | 71.45 | 85.78/93.33 | 65.64/45.15 | 1092/78/181/149 | 0.205 |
| histogram-balanced (fit) | 72.60 | 71.45 | 66.71 | 89.49/73.50 | 42.49/69.39 | 860/310/101/229 | 1.066 |
| histogram-unweighted (fit) | 81.20 | 68.48 | 70.02 | 85.63/91.20 | 59.45/45.76 | 1067/103/179/151 | 0.520 |

Fit times include fitting preprocessing, estimator and prediction; they exclude imports, CSV loading and process startup, so they are not comparable directly to complete CLI runtime. Existing estimator settings: logistic max_iter200/C1; histogram100 iterations,15 leaves,depth8,min_samples_leaf20,early_stopping=False; seed42. No parameter/threshold search.

Weighting is a demonstrated cause of the precision/recall tradeoff here: unweighted logistic removes270 false positives but adds92 false negatives; accuracy rises11.87 points and macro-F1 rises5.73, while BA falls2.40 and Yes recall falls27.88. Histogram shows the same direction:207 fewer false positives,78 extra false negatives; accuracy +8.60, macro-F1 +3.31, BA -2.97 and Yes recall -23.64 points. Balanced histogram does not fix the tradeoff, so family selection alone is not its cause. This does not prove a global weighting or selection defect: the optimization objective remains unknown, and neither weighting dominates all metrics.

## Datetime representation diagnostic (no date pipeline)

No equivalent controlled local evidence was found. REGISTRATION.json fixed one bike-demand-like synthetic diagnostic before fitting:2160 hourly rows, generator seed63761, chronological1620/540 split; target150+45*sin(2*pi*(hour-7)/24)+20*(weekday<5)+2*temperature+normal(0,8), temperature normal(18,5). Compare timestamp-string categorical plus temperature with numeric hour/weekday/month plus the same temperature. Targets/rows/split/model seed42/budget50 are identical. Deterministic date components replace the string; nothing is detected/expanded in production.

| Representation | RMSE | MAE | R2 | CLI seconds | Selected model |
|---|---:|---:|---:|---:|---|
| datetime-string | 33.382 | 29.008 | 0.070 | 7.32 | Ridge |
| date-components | 8.833 | 7.031 | 0.935 | 6.71 | HistGradientBoostingRegressor |

Components recover known periodic signal that unique timestamp categories cannot transfer to later rows. This is a deliberately constructed mechanism check, not evidence on actual Bike Sharing Demand or general date parsing. Both outputs are valid learned predictions with independently correct profiles and no emergency fallback. No production datetime change is included.

## Recommendation, scope and evidence

Retain this input-only repair for review. Best next classification experiment: one separately registered training-only threshold-calibration study of the existing balanced logistic model, keeping the current0.5 threshold as the incumbent. Freeze a precision/recall policy and unchanged-model comparison before fresh evaluations; report accuracy, BA, macro-F1 and minority precision/recall, retain alternatives when objectives conflict. The concentrated false positives justify studying the decision tradeoff; this single synthetic split cannot choose the threshold or establish generalization. Do not start it or promote unweighted models automatically. Prior rejected selector/capacity controls are not adopted.

Exact candidate files: src/unknown_dataset/contract.py; src/unknown_dataset/runner.py (one additive run field); tests/test_numeric_input.py; this report. Main and historical/evaluator files remain unchanged. No README rewrite. Production model/preprocessing/reliability source hashes match main.

Local-only evidence/scripts are under experiments/numeric-input-robustness: REGISTRATION.json, check.py, suite.py, finalize.py, source manifest, case audits/input hashes, paired metrics, date diagnostic inputs/outputs, test logs/XML and resumable progress. Those diagnostic scripts are separate from production and are not staged in this limited checkpoint. No downloads. Raw/reference/prediction arrays, caches and bulky artifacts remain local; originals and all historical evidence are preserved. Frozen source/config/test and input hashes were checked before calls. Numerical acceptance/runtime rules are engineering choices, not official grading metrics or certification of unknown hardware. Observed times do not guarantee all future inputs; numeric coercion may hide malformed feature text, which is why counts are exposed. Strict external consumers must allow the additive run field.
