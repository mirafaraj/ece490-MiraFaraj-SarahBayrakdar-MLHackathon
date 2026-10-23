# Unknown Dataset Hackathon

A CPU-only MVP for automatic classification, regression, and clustering on unfamiliar tables.
It produces ordered predictions, a structured training-data profile, and reproducibility metadata.

**Supplied specification:** the Unknown Dataset Hackathon student announcement is the complete
supplied specification. It specifies 18 private datasets, 85 quality points, 10 profiling points,
5 speed points, equal task contributions to quality, and a 60-second per-dataset CPU limit.
It leaves per-task metrics, normalization, failed-run penalties, concrete size limits, hardware,
and the exact machine-readable interface unspecified. The CSV/JSON interface, profile schema,
diagnostic metrics and resource rules below are our implementation interpretations. Compatibility
with an unspecified assessment interface has not been established.

## Setup

Tested with Python 3.12.14; exact dependencies are recorded in `requirements.lock`. Use Python
3.12 for reproduction. In a shell where `python` refers to Python 3.12, on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/evaluate.py --output experiments/baseline.json --seed 42
```

On Linux, substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe`.
In this workspace, the verified environment is `.venv312`; use
`.\.venv312\Scripts\python.exe -m pytest -q` and
`.\.venv312\Scripts\python.exe scripts/evaluate.py --output experiments/baseline.json`.
The initial `.venv` used Python 3.14 and hit Windows Application Control blocks in NumPy.
The original verified Python 3.12 environment reuses bundled NumPy/pandas. The pinned dependencies
also install in a new isolated local environment; final smoke evidence is kept separately.
This does not certify unspecified assessment hardware. CI for the present uncommitted candidate
has not been run; earlier checkpoint CI results apply to their recorded commits.
No network access or AI service is needed for prediction. Dependency installation requires internet
or a prepopulated wheel cache. The official Python version may require a separately verified lock.

## Run a dataset

```powershell
.\.venv\Scripts\python.exe main.py --input data/example --output outputs/example --time-budget 50 --seed 42
```

The CLI and public Python API default to `submission`. Explicit `--strategy baseline`,
`--strategy auto` and `--strategy routed` remain available. The opt-in routed strategy applies explicit
task-specific eligibility/metric rules, completes a baseline first, and preserves its predictions
while optional candidates run. `--strategy auto` remains available for earlier experiments.
See [MODEL_CARD.md](MODEL_CARD.md) for intended inputs, evidence and limitations.

The submission strategy delegates only by the declared task: classification uses
baseline; regression and clustering use auto. The following explicit command is equivalent
to omitting `--strategy`:

```sh
python main.py --input INPUT --output OUTPUT --strategy submission --time-budget 50 --seed SEED
```

The public Python entry point has the same default:

```python
from unknown_dataset.models import fit_predict

predictions, info = fit_predict(task, X_train, y_train, X_test, feature_types,
                                n_clusters=n_clusters, seed=42)  # strategy="submission"
```

Supply `None` for supervised-only arguments when clustering; supply the required k.
This default is an approved development choice, not a verified official scoring optimum.

This command and the input/output file names are implementation assumptions.
`INPUT`, `OUTPUT` and `SEED` are per-run parameters; omitting seed uses 42. Local frozen checks
use the saved case seed (42/314/2027 for real tables and 7001 for stress inputs).
Classification
baseline is a conservative choice under unknown grading metrics, not a proven optimum: the
existing real-data check found auto gained 3.90 balanced-accuracy and 11.78 minority-recall points,
while losing 1.57 accuracy and 1.45 macro-F1 points. Regression/clustering comparisons use
RMSE/MAE/R2 and ARI/AMI diagnostically; these are not verified organizer scoring metrics.

### Expensive-input safeguards (engineering assumptions)

At 2,500,000 or more training feature cells, regression uses a fixed Extra Trees preset:
16 trees, max_features="sqrt", depth 24, and at most 20,000 fitting rows. Classification uses
this preset only with at least 50,000 training rows under baseline/routed (including submission's
baseline mapping). Sampling is deterministic by seed and preserves observed classification
classes. Existing successful large-classification auto and wide baseline/routed paths are retained.
Auto skips histogram selection when there are at least 256 features and 2,500,000 training
feature cells, avoiding the diagnosed binning stall. No dataset names or reference labels are used.

For guarded forests, one fixed attempt receives at most 25 seconds and 65% of available worker
time. If it fails, the existing linear estimator is retried on the bounded sample. Emergency
mean/majority remains the final explicitly reported recovery. The parent uses one absolute
deadline and reserves up to 8 seconds (20% of the requested budget) for recovery validation and
complete output publication. Ordinary paths retain their existing output reserve. The 50-second
internal budget targets headroom below the announced 60 seconds; it does not certify unspecified
assessment sizes or hardware. Profiling always describes the full training data before sampling.

Our structured profile is `provisional-1`: training row/feature counts, declared/inferred types,
per-column missingness, cardinality, constants, exact duplicates, numeric min/max/mean/population
standard deviation/median, and target summaries/distributions with lexical classes preserved.
This schema is our interpretation of the announcement, not an organizer-provided schema.

### Local reliability verification

The repaired candidate passed 295 correctness tests. All 39 existing stress strategy runs
returned valid learned outputs with no emergency recovery (maximum 19.59 seconds), and all
108 existing real-data strategy runs preserved their historical prediction hashes and selected
models (maximum 28.61 seconds). Full-data profiles passed independent reconstruction; discrete
fields compare exactly and numeric summaries use relative tolerance 1e-10 / absolute 1e-12.
These measurements describe the recorded Windows CPU environment, not unspecified assessment
limits or hardware. The bounded forest trades some quality for headroom: on the 100k x50
classification fixture, accuracy was .9075 versus .9565 for the successful 96-tree diagnostic
run. No model-size sweep was performed. Detailed evidence and freeze/final-verification records
are kept separately in `experiments/reliability-preparation/`.

```powershell
.\.venv312\Scripts\python.exe main.py --input data/example --output outputs/example --strategy routed
.\.venv312\Scripts\python.exe main.py --input data/example --output outputs/example --strategy routed --router-config configs/router-accuracy.json
.\.venv312\Scripts\python.exe main.py --input data/example --output outputs/example --strategy routed --router-config configs/router-categorical.json
```

`configs/router.json` records provisional objectives and resource limits. Partial JSON overrides
are strictly validated. Classification supports explicit accuracy or balanced accuracy; regression
supports RMSE or MAE; eligible clustering uses bounded BIC diagnostics. These objectives are not
official contest metrics. Near-tie tolerance defaults to zero. A configured tolerance prefers
the simpler candidate only within that gap from the best validation score.

The categorical config explicitly enables an experimental single-feature count classifier.
Counts and training-prior smoothing are fitted inside the inner fitting fold; unseen categories
use its prior posterior, with the final decision rule following the configured objective.
This candidate is disabled by default and has not been promoted.

When optional work fails or is terminated, valid completed-baseline predictions are recovered
and marked `fallback_kind=completed_baseline`. Ordinary worker timeouts without a valid checkpoint
use marked emergency predictions; other ordinary failures without a checkpoint remain nonzero
errors. Guarded expensive-input work first tries its bounded forest, then existing linear recovery,
then marked emergency predictions if neither learned attempt succeeds.
`run.json` records the route, metric, exclusions, selected model, validation scores, timing and cause.

The input folder contains `metadata.json`, `train.csv`, and, for supervised tasks, `test.csv`.
Example supervised metadata:

```json
{
  "task": "classification",
  "target": "churn",
  "id_column": "customer_id",
  "feature_types": {"age": "numeric", "contract": "categorical"}
}
```

`task` is `classification`, `regression`, or `clustering`. Supervised training tables include
the target; test tables must omit it. Classification outputs are class labels, not probabilities.
The file adapter treats classification labels as opaque strings, preserving distinct labels
such as `01` and `1`. Direct Python modelling calls also preserve numeric class labels.
Clustering metadata omits `target`, declares integer `n_clusters`, and uses only `train.csv`.
Clustering outputs assign the training rows to the requested number of groups.

`feature_types`, if provided, covers every feature exactly once, excluding the target and optional
ID. Without it, training CSV dtypes determine feature types. IDs are excluded from modelling;
declared IDs must be nonmissing and unique. Test IDs may be omitted. Test features must match
training features; their column order may differ. Missing labels and nonfinite numeric inputs
are rejected. Numeric feature missing values are allowed. CSV parsing uses pandas' default NA tokens.
Both input tables must have at least one row and at least one feature.

Output files:

- `predictions.csv`: one `prediction` column, preserving test row order for supervised tasks
  and training row order for clustering.
- `profile.json`: per-feature type, missingness, cardinality, numeric summaries, constants,
  exact duplicate columns, aggregate counts, and optional target summaries.
- `run.json`: task, seed, budget, model choices, preprocessing decisions, and timing.

Malformed input and unrecovered errors return a nonzero exit code with an error on stderr.
Successful checkpoint, learned or emergency recovery can return zero; inspect `run.json` for
the recovery type and cause. A zero exit code alone does not establish predictive usefulness.

## Modelling and runtime

Training-fitted preprocessing normalizes categories, removes constants and duplicate columns,
imputes missing values with numeric missingness indicators, and bounds one-hot encoding to
32 categories per feature. Actual categories use a separate namespace from the missing-value
token. Supervised
models use Extra Trees; classification uses balanced class weights. Clustering scales numeric
features and uses K-means or MiniBatch K-means depending on input shape. Unknown categories
are handled by the fitted encoder. No evaluation labels enter training.

Profiles describe training features before model preprocessing. A profile constant has at most
one distinct nonmissing value, including all-missing columns. Model removal retains columns
whose missingness varies, since missingness can itself be informative. Duplicate profiling requires
equal values, missing positions, and dtype. Duplicate rows are retained.

The runner isolates model work in a spawned process and terminates it when its allocation expires,
reserving time for output. A clearly marked emergency fallback then uses the majority class,
training mean, or deterministic group assignments. Fallbacks prioritize a valid result and may have
poor quality; inspect `model.fallback` in `run.json`. Identical clustering points may require
deterministic splitting to occupy every requested group; the model records repairs.

The default internal budget is 50 seconds, leaving external headroom under the announced 60-second
limit. Imports before argument parsing, operating system scheduling, process termination, and file
I/O mean this is not a hard real-time guarantee. Benchmark the largest published cases on the
assessment environment. Native numerical libraries and estimators use one CPU thread.

## Evaluation and improvement

Current router integration and selector diagnostics are documented in
[reports/router_validation.md](reports/router_validation.md). The integrated suite passed
153 tests, followed by eight strengthened fitted-state invariance checks. The frozen router
comparison passed all 54 CLI runs with no fallbacks and stable source/config. Routed matched
auto quality on all 18 paired cases, with mean CLI time increasing from about 5.0 to 7.1 seconds.
This is an execution-control tradeoff, not a universal quality or speed improvement.
The offline stability study includes 148 valid direct-call runs, selection frequencies,
baseline losses and retrospective best-evaluated gaps. These direct-call timings are not CLI or
deadline evidence. Experimental categorical counts remain disabled and near-tie tolerance remains
zero; an exploratory positive tolerance worsened one development split.

```powershell
.\.venv312\Scripts\python.exe -u scripts/selector_stability.py --explore-tie --output experiments/selector_stability.json
.\.venv312\Scripts\python.exe -u scripts/evaluate.py --suite router --seeds 42 8081 --strategies baseline auto routed --output experiments/router-comparison.json
.\.venv312\Scripts\python.exe scripts/summarize_comparison.py experiments/router-comparison.json --reference auto --candidate routed --output experiments/router-vs-auto.json
```

Seed 8081 was predeclared confirmation for the frozen router implementation; after its results
are inspected it becomes development evidence for later changes. Do not retune against a report
and continue describing that same report as untouched confirmation. The official metric gate
remains unresolved. Historical default statements in frozen reports describe their recorded versions;
the current default is the task-specific submission mapping documented above.

Latest targeted follow-up: `reports/clustering_improvement.md` documents the optional clustering
restart fix. It restores the observed noisy-data ARI regression (0.6337 to 1.0000), while fresh
confirmation across ten cases found four wins, five ties, and one small loss versus baseline.
All 20 CLI runs were valid without fallbacks. Baseline remains explicitly selectable;
submission now uses auto for clustering and regression.

Current review and expanded comparison results are documented in `reports/RESULTS.md`.
The frozen broad comparison passed all 48 CLI runs with zero deadline fallbacks. Supervised
candidates improved several cases, but balanced accuracy and ordinary accuracy can move in
opposite directions. Clustering also showed an observed regression. Submission's task mapping
is based on development evidence; official metric compatibility and broader generalization remain unverified.

Independent review reports are in `reports/review_data.md`, `reports/review_runtime.md`, and
`reports/review_evaluation.md`. Regression fixes preserve lexical labels and missingness signals,
avoid categorical token collisions, clear previous generated outputs at the start of a CLI run,
and stage all result files before publication. Publication failures remove partial generated results.

The original model family remains available as explicit `--strategy baseline`, with these
correctness fixes applied. Use `--strategy auto` for experimental training-only supervised selection
among Extra Trees, linear models, and histogram boosting; clustering compares spherical/full mixture
geometry for eligible small numeric tables. Selection metadata is written to `run.json`. These
development metrics and single-holdout choices may differ from the eventual official scoring rules.

Run the expanded paired benchmark using the verified local environment:

```powershell
.\.venv312\Scripts\python.exe -u scripts/evaluate.py --suite broad --seeds 42 314 --strategies baseline auto --output experiments/broad-comparison.json
.\.venv312\Scripts\python.exe scripts/summarize_comparison.py experiments/broad-comparison.json --output experiments/comparison_summary.json
```

The broad suite contains 12 families: breast cancer, wine, imbalanced classification,
high-cardinality classification, diabetes, linear regression, nonlinear regression,
mixed/missing regression, noisy blobs, anisotropic clusters, moons, and mixed clusters.
Breast cancer, wine, and diabetes are bundled offline real datasets. The remaining cases are
synthetic. Seed 42 is development; seed 314 is confirmation. For real datasets these are different
splits of the same source data, not independent new datasets. Source/input hashes verify paired
comparisons, and the evaluator fails if source code changes while it is running.

The evaluator independently verifies provisional profile semantics. Classification metrics encode
opaque labels consistently; cluster checks use each dataset's declared k. Candidate implementations
never receive benchmark references. Reports distinguish model errors, invalid outputs, and deadline
fallbacks. The historical three-case `experiments/baseline.json` remains unchanged.

`scripts/evaluate.py` generates reproducible mixed synthetic tables with missing values, nonlinear
signals, irrelevant features, constants, duplicates, and unseen test categories. It runs the actual
CLI in subprocesses with a 60-second external timeout and reports classification accuracy/balanced
accuracy, regression RMSE/R2, clustering adjusted Rand index, dummy supervised baselines, output
validity, and wall time. These three synthetic cases are smoke benchmarks, not the nine official
practice datasets or a reliable estimate of hidden performance.

Keep `experiments/baseline.json` as the first reference and write new runs to separate filenames.
Change one modelling decision at a time and compare per-task quality, failures, and runtime on fixed
splits/seeds. Add additional held-out datasets/seeds before promoting improvements. Adapt metrics
and aggregation to the official scoring rules before optimizing contest points. Candidate improvements
include histogram gradient boosting, linear models, better categorical weighting for clustering,
and bounded validation-based model selection.

Verified initial run (seed 42): **34 tests passed**. All three synthetic cases returned valid
outputs: classification accuracy 0.8467 (majority baseline 0.6800), regression RMSE 1.7695
(mean baseline 3.5627), clustering adjusted Rand index 1.0000. CLI wall times were 5.04,
4.91, and 4.39 seconds respectively. These measurements do not establish worst-case runtime
or official assessment performance. See `experiments/baseline.json` for full metadata.

## Project organization and phases

`main.py` is the entry point. `contract.py` owns file adapters, `profiling.py` descriptive statistics,
`preprocessing.py` training-fitted transformations, `models.py` estimators, and `runner.py` process
orchestration. `tests/` covers meaningful edge cases and contract behavior. See `PHASES.md` for
implementation gates, agent ownership, and remaining official-submission work.

Names identified by the supplied repository: **Mira Faraj and Sarah Bayrakdar**; any third member's
name remains to be supplied. The existing GitHub repository is private. The announcement requires
a public repository at submission; pushing or changing visibility requires separate user approval.
Retain runnable source, pinned dependencies, entry point, lightweight tests, reusable evaluation
scripts and concise reports. Raw evaluation data, references, caches and bulky outputs are not
required by the announcement and remain local unless separately approved. Record the submitted
commit once approved; exact assessment-interface and hardware compatibility remain unresolved.
