# Unknown Dataset AutoML

## Intended use and inputs

CPU-only development baseline for unfamiliar tabular classification, regression and clustering.
The file adapter accepts metadata.json and train.csv; supervised tasks also supply test.csv.
Metadata declares task, target for supervised tasks, optional ID and feature types, and clustering k.
Rows retain their input order. Classification labels are opaque strings in CSV; regression labels
must be finite numeric values. CSV/JSON is the supported file interface; mostly-zero features do
not imply support for sparse file formats. See README.md for the complete provisional contract.

The student announcement is the complete supplied specification. It leaves the machine-readable
contract, profile schema, per-task scoring formulae, concrete size limits and CPU details
unspecified. The repository interface and diagnostic metrics are implementation assumptions.
**Official interface compatibility is unverified.** No clinical,
financial, or production decision use is claimed.

## Preprocessing and selection

Training-fitted preparation handles missing values and numeric missingness indicators, category
namespaces, bounded one-hot encoding, constants and exact duplicate columns. Scaling is used for
linear/distance-based models. Duplicate rows are retained; IDs and supervised targets are excluded
from features. Unknown categories are handled without refitting on test inputs.

The default remains baseline: Extra Trees classification/regression and scaled K-means/MiniBatch
K-means. Auto is experimental: supervised linear/tree/histogram candidates compete on an inner
training holdout; eligible numeric clustering tables compare spherical/full covariance density
models. Its generic/centroid restart fix addresses a reproduced local-optimization failure.

The opt-in routed strategy uses a training-only summary and versioned eligibility/metric rules.
It checkpoints complete baseline predictions before optional model work and reuses that baseline
if selected. Explicit accuracy/balanced-accuracy and RMSE/MAE objectives control supervised fitting.
Near-tie tolerance defaults to zero; a configured positive tolerance prefers a simpler candidate only
within that metric gap from the best validation score. No positive default was promoted.

The experimental single-feature categorical classifier is disabled by default. It uses
`(category_class_count + alpha * fitting_class_prior) / (category_count + alpha)`; unseen categories
use the fitting prior. Missing and genuine category tokens are distinct. Inner fitting folds own
all counts and priors; feature/smoothing choices use inner validation only. Explicit balanced
decisions may differ from posterior argmax while the reported probabilities retain prior smoothing.
None of these features replaces the baseline default.

Explicit `submission` delegates classification to baseline and regression/clustering to auto;
the CLI and API defaults remain baseline. This task mapping is conservative development judgment
under unknown grading metrics; classification baseline's measured balanced-accuracy disadvantage
remains relevant. See README for fixed expensive-input forest/sample rules, histogram eligibility,
absolute worker deadline, existing linear recovery and full-data profiling. These guards address
local stress failures, not unspecified maximum assessment sizes. No new model family is added.

Reliability candidate checks: 295 tests passed; 39/39 stress runs produced learned outputs
without emergency recovery; 108/108 ordinary real-data runs retained exact prediction hashes
and selected models. Independent full-data profile checks passed. The fixed bounded forest's
large-classification accuracy .9075 is lower than the successful 96-tree diagnostic run's
.9565; this is a documented quality/runtime tradeoff. Frozen verification results are recorded
separately and do not establish organizer-environment or official-score compatibility.

Current development metrics are provisional: classification accuracy/balanced accuracy,
regression RMSE/R2 (and explicit alternative objectives where configured), clustering ARI for
offline reference-labelled evaluation and BIC for optional unsupervised selection. Better balanced
accuracy can reduce ordinary accuracy; better density fit can disagree with reference groups.
Neither relationship establishes an official-scoring gain.

## Reproducibility and resources

Previously verified local environment: Windows, Python 3.12.14, exact versions in requirements.lock,
seeded estimators/splits, one numerical/estimator CPU thread. Verification used .venv312 and bundled
NumPy/pandas. The pinned dependencies also install in a new isolated local environment; final
entry-point smoke results are recorded separately. Assessment-hardware compatibility and CI for
the present uncommitted candidate remain unverified. No AI/network service is needed for inference.

The CLI default internal budget is 50 seconds with an output reserve; the announced external limit
is 60 seconds. A parent process can terminate model work. Python startup, imports, scheduling and
I/O mean this is not a hard real-time guarantee. Largest-size and peak-memory checks remain pending.
Expanded/repeated-split model evaluation runs offline, outside prediction-time execution.

## Measured evidence and limitations

Historical evidence: reports/RESULTS.md and reports/clustering_improvement.md. The latter records
68 passing tests and 20 valid clustering CLI runs without fallbacks; compared with baseline, four
fresh-seed cases improved, five tied and one slightly regressed. Reported 5.84–7.35 second CLI
times were partly contended and are not speed-improvement or maximum-size claims.

Historical reports and source snapshots refer to their recorded versions, not every later edit.
Small synthetic families and repeated splits of bundled real tables do not demonstrate hidden-test
superiority. Previously inspected confirmation results become development evidence for later changes.
Moons/circles, rare groups, high cardinality and selector instability remain relevant limitations.

Current emergency outputs are explicitly marked majority class, training mean or round-robin groups;
they prioritize valid output and may have poor quality. Routed timeout/crash/error recovery returns
a validated completed baseline when available, marked `fallback_kind=completed_baseline`; optional
refit failures are also counted. Corrupt checkpoints are rejected. Invalid inputs fail
with nonzero status, and stale/partial generated artifacts are cleaned up.

Historical implementation verification: **153 tests passed**, including real subprocess recovery,
contract/parity checks, fold isolation, lexical/fractional labels, prediction-row permutation,
test-feature invariance of fitted state/selection, and evaluation-label invariance of predictions.
Only expected identical-point clustering warnings were observed.

Offline development study: **148 valid direct-call runs**, eight supervised families and two
overlapping outer splits. Routed selection had ten wins, five ties and one loss versus baseline;
selection frequencies and empirical gaps are in experiments/selector_stability.json. The categorical
variant was not selected on its two measured splits. A positive exploratory tie rule changed one
selection, reducing balanced accuracy by 0.0389 with little timing benefit; zero remains default.
These are direct-call timings, not CLI/deadline certification, fresh confirmation, or new datasets
for every split. See reports/router_validation.md for complete current-source evidence.

Frozen current-source CLI comparison: **54/54 valid**, no fallbacks, nine families/two seeds/three
strategies, matching input hashes and stable source/config. Routed tied auto quality in all 18
paired cases but mean CLI time increased from 4.973 to 7.127 seconds; routed ranged 6.399–9.546
seconds. Against baseline on confirmation, routed had four wins/four ties/one loss across tasks,
using provisional balanced accuracy/RMSE/ARI. One clustering ARI declined 0.3838 to 0.2962, and one
classification case improved balanced accuracy while reducing ordinary accuracy. No default or
positive tie policy was promoted. After this run, eight strengthened fitted-state/evaluation-label
invariance tests passed; production source was unchanged.

## Claims and external references

Measured results must link to reports with source/config/input hashes. Runtime observations are
limited to the recorded environment. Design settings and untested limits are assumptions, not
measurements. Public Stories Coffee repository techniques motivate experiments only; their reported
results were not independently reproduced here, do not establish superiority, and supply no reason
to change clustering algorithms. Official alignment and fresh reviewed evidence are required before
changing the baseline default.
