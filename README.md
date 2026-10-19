# Unknown Dataset Hackathon

A CPU-only MVP for automatic classification, regression, and clustering on unfamiliar tables.
It produces ordered predictions, a structured training-data profile, and reproducibility metadata.

**Status:** development baseline. The official kickoff contract, nine practice datasets, scoring
formulae, and assessment environment are not available in this repository. The local contract
and evaluation metrics below are provisional; official compatibility has not been verified.

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
The verified Python 3.12 environment reuses bundled NumPy/pandas; installation into a completely
clean environment and the CI workflow remain separate verification steps.
No network access or AI service is needed for prediction. Dependency installation requires internet
or a prepopulated wheel cache. The official Python version may require a separately verified lock.

## Run a dataset

```powershell
.\.venv\Scripts\python.exe main.py --input data/example --output outputs/example --time-budget 50 --seed 42
```

The default remains `--strategy baseline`. The opt-in `--strategy routed` applies explicit
task-specific eligibility/metric rules, completes a baseline first, and preserves its predictions
while optional candidates run. `--strategy auto` remains available for earlier experiments.
See [MODEL_CARD.md](MODEL_CARD.md) for intended inputs, evidence and limitations.

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
and marked `fallback_kind=completed_baseline`. Weak emergency predictions are used only for
timeouts with no valid checkpoint. Other failures without a checkpoint remain nonzero errors.
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

Malformed input and model errors return a nonzero exit code with an error on stderr. The CLI
does not silently turn unexpected model failures into successful predictions.

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
and baseline default remain unchanged.

Latest targeted follow-up: `reports/clustering_improvement.md` documents the optional clustering
restart fix. It restores the observed noisy-data ARI regression (0.6337 to 1.0000), while fresh
confirmation across ten cases found four wins, five ties, and one small loss versus baseline.
All 20 CLI runs were valid without fallbacks. The baseline default remains unchanged; use
`--strategy auto` to exercise the improved experimental selector.

Current review and expanded comparison results are documented in `reports/RESULTS.md`.
The frozen broad comparison passed all 48 CLI runs with zero deadline fallbacks. Supervised
candidates improved several cases, but balanced accuracy and ordinary accuracy can move in
opposite directions. Clustering also showed an observed regression. The default remains baseline;
`--strategy auto` is explicit and experimental pending official metrics and broader evidence.

Independent review reports are in `reports/review_data.md`, `reports/review_runtime.md`, and
`reports/review_evaluation.md`. Regression fixes preserve lexical labels and missingness signals,
avoid categorical token collisions, clear previous generated outputs at the start of a CLI run,
and stage all result files before publication. Publication failures remove partial generated results.

The original model family remains available as `--strategy baseline` (the default), with these
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

Team names: **to be supplied before submission**. No GitHub repository has been created or published.
Before submission, add names, adapt and pass the official checker, verify all nine practice datasets
and size limits, install from scratch on the official environment, and record the submitted commit.
