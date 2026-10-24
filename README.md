# Unknown Dataset Hackathon

**Team: Mira Faraj and Sarah Bayrakdar.**

One CPU-only program automatically handles unfamiliar tabular classification,
regression and clustering datasets. It returns ordered predictions, a structured
training-data profile and run metadata. Prediction requires no network or AI service.

## Install

Use Python **3.12** (locally verified with 3.12.14). From the repository root,
install the exact dependencies in `requirements.lock`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
```

Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

Run commands below with that environment's Python. For example, replace `python`
with `.\.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux.
Dependencies must already be installed when assessment begins.

## Standard entry point

```sh
python main.py --input INPUT_DIRECTORY --output OUTPUT_DIRECTORY
```

`INPUT_DIRECTORY` must be an existing folder following the layout below; it is
not a literal example folder shipped with the repository. Output directories are
created as needed. The defaults are `--strategy submission --seed 42 --time-budget 50`.
No extra strategy flags, local wrappers or environment flags are needed.

`submission` delegates **classification, regression and clustering to existing auto**.
Explicit `baseline`, `auto` and `routed` remain available for reproducibility.
Native numerical libraries and estimators use one CPU thread.

## Input layout (our interpretation)

Classification and regression input folders contain:

```text
metadata.json
train.csv
test.csv
```

Training rows include the target; test rows must omit the target. The grader's
private test answers are never required or read. A single unsplit CSV must be
prepared into this layout outside the prediction program. This program does not
automatically split a labelled dataset or score its own test predictions.

Example classification metadata:

```json
{
  "task": "classification",
  "target": "churn",
  "id_column": "customer_id",
  "feature_types": {"age": "numeric", "contract": "categorical"}
}
```

For regression, use `"task": "regression"` and name its numeric target.
`id_column` is optional. IDs are excluded from features and, when supplied, must
be nonmissing and unique. Test IDs may be omitted. Classification labels are
opaque strings, preserving distinct CSV labels such as `01` and `1`.

Clustering input contains `metadata.json` and `train.csv` only. No labels or target
column are supplied; outputs assign those training rows to the requested groups.
For example:

```json
{
  "task": "clustering",
  "n_clusters": 3,
  "feature_types": {"annual_income": "numeric", "spending_score": "numeric"}
}
```

The requested group count must be an integer between 1 and the number of rows.
`feature_types`, when supplied, must describe every feature exactly once using
`numeric` or `categorical`, excluding target and ID. Otherwise training CSV dtypes
infer the types. Train/test feature names must match; order may differ. Both
supervised tables must contain at least one row and one feature.

Missing numeric features are allowed. In explicitly declared numeric features,
whitespace and unparseable entries become missing; `run.json` records coercion
counts. An entirely unparseable feature becomes all missing and can be removed by
preprocessing. This conversion does not apply to targets, IDs or categorical
features. Missing targets, infinite numeric values and malformed metadata are
rejected. CSV parsing uses pandas' default missing-value tokens.

## Outputs and profiling

- `predictions.csv`: one `prediction` column, in original test-row order for
  supervised tasks or original training-row order for clustering.
- `profile.json`: structured full-training-data profile before model sampling.
- `run.json`: selected model, selection scores, seed, runtime, coercion counts,
  preprocessing information and any recovery reason.

Profile schema `provisional-1` includes row/feature counts, feature types,
missingness, cardinality, constant and exact duplicate columns, numeric
min/max/mean/population standard deviation/median, categorical top-20 values with
full-training counts and omitted counts, and supervised target summaries.
Clustering also records the **requested** cluster count. A profile constant has
at most one distinct nonmissing value, including all-missing columns. Duplicate
profiling requires equal values, missing positions and dtype.

A successful exit code confirms output publication, not high predictive quality.
Inspect recovery metadata too. Accuracy and errors require external reference
answers, so they are calculated by an offline evaluator, not the submitted
prediction command. Malformed input and unrecovered errors return nonzero status.

## Approach

All model preparation is fitted on training data, with inner selection preparation
fitted only on its fitting partition. IDs and supervised targets are excluded.
Preparation handles missing values, numeric missingness indicators, constants,
exact duplicate columns, categorical encoding capped at 32 categories per feature,
and scaling where needed. Unseen categories use the fitted encoder. Duplicate
rows are retained.

Classification auto compares Extra Trees, logistic regression and histogram
boosting with balanced class weights, selecting by training-holdout **balanced
accuracy**. Regression compares Extra Trees, a linear model and histogram boosting,
selecting by negative RMSE. Successful selected models are refitted on full training
data unless an existing resource guard explicitly uses a bounded sample.
Clustering uses scaled K-means/MiniBatch K-means and eligible bounded Gaussian
mixture alternatives, with training-only unsupervised selection. No test answers
enter modelling or candidate selection.

The optional routed strategy preserves a completed baseline while attempting
optional work. Experimental date expansion, soft-weight candidates and clustering
noise-weighting studies are **not** adopted by this submission.

## Runtime and recovery

The announcement sets a **60-second per-dataset CPU limit**. Our default internal
budget is 50 seconds, with output reserves. The parent isolates model work in a
spawned process and enforces its worker deadline. Existing general input-cost
guards use bounded forests/samples and existing linear recovery. The fixed mixed
classification rule is documented in
[RESOURCE_RULE.md](experiments/classification-runtime-repair/RESOURCE_RULE.md).

Ordinary supervised auto now saves validated predictions from its already-fitted
winning selection model before the full-data refit. A successful full fit takes
precedence. If later work times out or fails and the checkpoint is valid, it
recovers that learned output as `fallback_kind=selection_model`, with
`fitting_scope=selection_partition`, actual `training_rows`, and
`available_training_rows`. This does not claim a completed full-data fit.
Routed checkpoints retain `fallback_kind=completed_baseline`.

If no usable learned checkpoint exists, an explicitly marked emergency majority,
mean or deterministic group output remains the final safety net. Those outputs
can be low quality. Corrupt checkpoints are rejected. Profiling still describes
all training rows. Startup, scheduling, memory and output costs mean runtime on
unknown hardware cannot be guaranteed.

## Verification and limitations

The final repair's development suite passed **355 tests**. All **120 saved CLI
configurations** retained predictions, model selections and profiles, including
36 default real-data cases and 39 stress strategy combinations. The stress maximum
was 26.60 seconds; the maximum among those 120 was 41.72 seconds. Another 66
historical real-data strategy combinations were not rerun. These counts describe
the recorded validation, not every possible dataset or official grading results.

Independent fresh full-data CLI checks reproduced Bike Sharing recovery at
**47.50s, RMSE 98.87, R2 0.798**, and Metro Traffic at
**47.58s, RMSE 1909.95, R2 0.073**. Both use 3,000 fitting rows with full-data
profiles. They improve on emergency means, but Metro remains weak. Both pass the
announced 60 seconds while missing our provisional 45-second headroom target.
Independent focused recovery testing passed 46 tests. See
[the repair report](experiments/auto-selection-recovery/REPORT.md).

Class balancing can improve rare-class recall while increasing false positives
and reducing ordinary accuracy or precision. Clustering can deteriorate with
irrelevant features, noisy missingness or non-centroid-shaped groups. Sampled
selection and repeated reuse of development datasets limit generalization claims.
Date-like strings currently receive ordinary categorical handling; automatic date
expansion is not enabled. Some inputs can still exhaust resources before any
usable learned checkpoint exists.

## Specification and assumptions

The supplied student announcement specifies 18 private datasets, **85 quality,
10 profiling and 5 speed points**, equal task contributions to quality, and the
60-second limit. Speed credit depends on useful performance. It mentions further
kickoff materials, but those materials were not supplied to this implementation.
Exact task metrics, score normalization, failure penalties, concrete size limits,
CPU details, input/output contract and profile schema therefore remain unverified.

CSV/JSON layout, filenames, metadata fields, train/test separation, profile schema,
run command, strategy mapping, diagnostic metrics, resource guards and internal
budgets are our documented interpretations, not additional official requirements.
No single accuracy, balanced accuracy, F1, RMSE, R2, ARI or AMI value is assumed to
be the official grading metric. No estimated score is an official grade.

The repository is public. Raw evaluation datasets, private references, full local
prediction dumps, caches and environments are not needed for submission and remain
local. Small synthetic test fixtures and historical concise evidence are versioned.
The submitted commit is frozen at the deadline; historical reports describe their
own recorded versions, not necessarily the final one.

## Reproduce checks

```sh
python -m pytest -q
python scripts/evaluate.py --strategies baseline auto routed --output outputs/local-evaluation.json
```

The evaluator creates local synthetic/bundled-data comparisons; it is not the
organizer's private checker or the nine promised practice datasets. Reference-based
metrics belong to offline evaluation. Use new output locations to preserve history.
GitHub Actions installs the pinned lock and runs tests plus CLI evaluation on
Linux/Python 3.12. CI passed for the preceding main checkpoint; check the final
submitted commit's Actions result separately.

`main.py` runs directly from the checkout without package installation. To import
`unknown_dataset` from a separate Python script, first install the local package
using the already pinned environment: `python -m pip install --no-deps -e .`.
See [MODEL_CARD.md](MODEL_CARD.md) for a compact model description.
