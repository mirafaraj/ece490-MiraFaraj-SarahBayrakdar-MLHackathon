# Model card: Unknown Dataset Hackathon

Team: Mira Faraj and Sarah Bayrakdar.

## Intended use

Automatic CPU-only classification, regression and clustering on unfamiliar tables.
The standard entry point is `python main.py --input INPUT --output OUTPUT`.
See README.md for installation and the provisional metadata/CSV contract.
Prediction needs no network, AI service or evaluation labels. No clinical,
financial or production-decision deployment is claimed.

## Current model

Default `submission` delegates all tasks to auto. Training-only holdout selection
compares balanced Extra Trees/logistic/histogram classifiers by balanced accuracy,
and tree/linear/histogram regressors by negative RMSE. Clustering compares scaled
centroid models and eligible bounded mixture alternatives without reference labels.
Preparation handles missingness, bounded categorical encoding, scale differences,
constants and duplicate columns, fitting transformations within training partitions.
IDs and supervised targets are excluded from features; duplicate rows are retained.

Explicit baseline and routed strategies remain available. Experimental date parts,
soft weights and clustering noise weighting are not included in the default.
Numeric input coercion applies only to declared numeric feature columns and is
logged. Full-training profiles include categorical frequencies and requested k.

## Reliability

The 50-second default internal budget reserves output time below the announced
60-second limit. Cost guards use bounded existing estimators/samples. Ordinary
supervised auto checkpoints the winning already-fitted selection model before
full refitting. Full-fit success takes precedence; recovered selection predictions
report actual fitting rows and `fallback_kind=selection_model`. Routed recovery
retains completed-baseline metadata. Corrupt checkpoints are rejected. With no
usable learned checkpoint, marked emergency mean/majority/group outputs remain.
They can have poor predictive quality. Profiling always covers full training data.

## Evidence

The repair passed 355 development tests; 120 saved CLI configurations matched
predictions/selections/profiles, including 36 default real-data cases and 39 stress
combinations. Another 66 historical strategy comparisons were not rerun.
Independent focused recovery testing passed 46 tests. Fresh full Bike and Metro
runs reproduced learned recovery at 47.50/47.58 seconds, with R2 0.798/0.073.
Both used 3,000 fitting rows and missed the provisional 45-second target while
remaining below 60 seconds locally. These are measured diagnostics, not official
scores or a guarantee for unseen inputs. Details:
[repair report](experiments/auto-selection-recovery/REPORT.md).

## Limitations and assumptions

Official per-task metrics, normalization, interface/schema, exact size limits and
hardware were not supplied. The README distinguishes announcement requirements
from implementation choices. Balanced weighting can increase false positives;
regression remains weak on some real temporal data; clustering is sensitive to
irrelevant columns and noisy missingness. Date strings are ordinarily categorical.
Finite development cases and reused datasets do not prove broad generalization.
Some inputs may time out before a learned checkpoint is available.

Pinned requirements, seeded splits/estimators and single-thread execution support
reproduction; timing varies with hardware and scheduling. No universal quality or
hard real-time guarantee is claimed. Historical reports remain preserved and apply
to their recorded versions. Only the final verified commit should be submitted.
