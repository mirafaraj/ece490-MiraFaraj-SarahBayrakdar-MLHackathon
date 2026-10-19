# Targeted first-batch integration report

This batch seeks improvements to unseen-table performance while retaining the
working baseline. All changes are optional. The official scoring rules remain
unknown, so accuracy, balanced accuracy, macro F1, ARI and AMI are provisional
scenarios; these development measurements cannot establish official score gains.

## Findings and decisions

The exact inspected classification loss is reproduced with model seed 8081:
balanced logistic regression has BA 0.83086 and minority recall 0.81818;
unweighted logistic has BA 0.72542 and recall 0.48485; unweighted Extra Trees has
BA 0.65152 and recall 0.30303. Along this path, changing weights accounts for
10.54 BA points and changing the selected family for another 7.39, totaling
17.93 points. This decomposition depends on the path: unweighting trees actually
improves their BA here. Both weighting and model selection matter, and their
effects interact. See classification.md for all four corners and other metrics.

Classification now separates class weighting, objective and one/two holdouts.
The same three estimator families are tested with balanced/unweighted fitting,
and candidate decisions use training-fold predictions only. An expanded pool
shares the existing total selection allowance. Incomplete fold results are
excluded, rather than compared with fully evaluated candidates. The saved
completed baseline remains the recovery target after optional refit failure.

On 12 small development cases, balanced weighting with balanced-accuracy
selection improved as follows when moving from one to two holdouts:

| Metric | One holdout | Two holdouts |
| --- | ---: | ---: |
| Ordinary accuracy | 0.81056 | 0.81167 |
| Balanced accuracy | 0.68129 | 0.69095 |
| Macro F1 | 0.65357 | 0.66039 |
| Mean recall across minority classes | 0.55317 | 0.58057 |
| Mean gap to retrospectively best evaluated BA in the same portfolio | 0.01077 | 0.00111 |

This is promising selector evidence, not a uniform win. The worst two-versus-one
holdout changes were -0.03333 accuracy, -0.01333 BA and -0.02231 macro F1.
Unweighted accuracy selection with two holdouts achieved higher ordinary
accuracy (0.85944), but lower BA (0.64167). Enlarging the portfolio to both
weightings did not reliably improve performance. No objective policy is promoted.

A severe remaining weakness is 95%-majority multiclass data: in one case the
BA-selected weighted linear model scored 55.33% accuracy versus 94.67% for the
weighted tree baseline, while improving BA from 0.3333 to about 0.4378. Another
case still has zero minority recall. Alternative metrics, class supports and
worst-case losses must accompany every future quality claim. Selection frequency,
all fold metrics and retrospective gaps are retained in the raw results.

Clustering interventions reproduce the original missing-axis failure and
establish that missing-indicator representation contributes to it. Full covariance
alone is not the explanation: switching to diagonal did not fix seed8081.
Density-based selection can further damage an already weak representation.
The explanation that standardized binary indicators dominate geometry and
density fitting remains a mechanism hypothesis rather than a full decomposition.

| Clustering policy | Mean ARI | Mean AMI | Decision |
| --- | ---: | ---: | --- |
| Current | 0.48195 | 0.51501 | Preserve |
| No numeric missing indicators | 0.74478 | 0.74531 | Promising optional candidate |
| K-means only | 0.51323 | 0.54539 | Retain ablation and safeguard |
| Diagonal covariance | 0.45101 | 0.48673 | Reject blanket replacement |

The indicator-free candidate improved 61 cases, lost eight and tied 21. Its worst
ARI loss was 0.32086; blanket K-means/diagonal alternatives lost as much as
0.52051 on correlated groups. All 360 outputs were valid, and 20 repeated
configurations reproduced exactly. Detailed per-family evidence is in
clustering.md. No combined clustering policy was tested.

## Runtime, quality and recovery

The fixed protocol completed 54 unique serial CLI runs: two development shapes
(96 x 8 and 660 x 110 training rows/features), budgets 10/25/50 seconds, three
repeats, and randomized baseline/auto/routed order within each paired block.
All 54 outputs were valid; maximum external wall time was 27.078 seconds. Four
runs recovered a completed baseline and two used emergency predictions. These
small development fixtures do not prove unpublished maximum-size compliance.

The bounded harness saved a 39-run primary artifact and successive 9- and 6-run
suffix artifacts. No completed row was repeated. Their aggregate verifies exact
54-row planned inventory, shared production/configuration hashes and identical
fixture bytes. The three timing segments took 576.47 seconds collectively. This
is the offline study duration, not a prediction-time budget.

| Wider fixture, budget | Strategy | Mean BA | Mean external seconds | Outcome |
| --- | --- | ---: | ---: | --- |
| 25 seconds | Auto | 0.75616 | 21.969 | Two strong completions, one emergency fallback |
| 25 seconds | Routed | 0.50000 | 22.531 | Three completed-baseline recoveries; zero minority recall |
| 50 seconds | Auto | 0.88424 | 21.719 | Three strong completions |
| 50 seconds | Routed | 0.88424 | 26.203 | Same predictions as auto in all three repeats |

This confirms a deadline-quality weakness: a valid recovered baseline need not
be a useful model for the task. Conversely, one natural 10-second routed recovery
on the smaller fixture retained accuracy 0.850 and BA 0.77429.

A separate controlled failure used genuinely fitted predictions: checkpoint
recovery retained accuracy 0.89167, BA 0.84065, macro F1 0.85713 and minority recall
0.72727, versus emergency 0.725, 0.5, 0.42029 and 0.0 respectively. It demonstrates
potential recovery benefit without estimating natural timeout probabilities.

Checkpoint writes average roughly 0.01 seconds. Parent setup not accounted for
by the named phases is roughly 2.4 seconds higher for routed than baseline/auto;
the code loads routing, preprocessing and sklearn during parent configuration
loading. This is a strong import-overhead hypothesis requiring an explicit timing
ablation, not a proven attribution of the entire gap. Keep the safeguards while
testing lazy training-only imports in the next batch. No overhead optimization
was promoted or applied in this batch.

Process-tree memory sampling was unavailable because psutil is not installed.
No dependencies were changed. Timing repeatability and prediction repeatability
are distinct: forced deadline fallbacks can alter predictions in tight budgets.
All metrics, validity, fallback kind, phase coverage and limitations are reported
together in runtime.md and its raw artifacts.

## Evidence boundaries and reproducibility

Classification uses 12 bounded rotating-factor cases, two exact repeat cases,
and an inspected historical diagnostic. These are not 12 independent families.
The exhaustive offline decomposition differs from prediction-time model search;
all six variants/two folds took at most 4.73 seconds of measured inner work per
development case, but maximum-size deadline compliance remains unverified.
The first run failed at temporary-file storage after completing fits; a corrected
workspace temporary directory and incremental result saves allowed completion.
The main successful retry took 91.58 seconds collectively. Both repeats matched
candidate predictions, inner/outer metrics and selection decisions exactly.

Clustering uses six mechanisms, five missingness conditions, three development
seeds and four isolated policies at 240 rows, taking 22.50 seconds collectively.
Historical seed42/8081 results are inspected diagnostics. All references stay in
offline harnesses; reference metrics are never used for production selection.

Baseline models, regression estimator definitions, shared preprocessing, default
configuration file, dependencies and the evaluator's framework are preserved.
New controls and observational timing metadata are opt-in. Test and frozen-source
verification results are recorded below.

## Final integration and preservation checks

- The final integrated correctness run passed **153 tests in 80.07 seconds**.
  It covers the new controls, configuration validation, lexical labels, contract,
  missingness, required groups, leakage invariance, baseline reuse and regression
  behavior. It also passes the strengthened 96-row CLI timing-on/off test with
  active optional selection and identical prediction bytes.
- A separate focused runtime/checkpoint/runner run passed **35 tests**, including
  actual spawned-worker deadline, crash, invalid-result and checkpoint recovery
  checks. Those already-passing legacy files were not rerun in integration;
  their strengthened CLI test was repeated. Together the runs cover 187 distinct
  test cases. Two integration warnings arise on intentionally identical-row
  clustering fixtures; required-group repair still passes.
- **18 frozen-source comparisons matched exactly** for predictions, selected
  candidate and validation scores/BICs against the pre-batch checkpoint. They
  cover baseline/auto/routed across all three tasks, with additional sparse-linear,
  heavy-tail and missing-product regression probes. This is measured default
  compatibility on fixtures, not a proof of identical deadline outcomes on every
  possible table.
- All **109 original files outside the five intended source edits** are preserved
  byte-for-byte, including all **33 evaluator files**, historical reports and
  benchmarks, default configuration, dependencies and immutable baseline models.
  `git diff --check` passes. The local HEAD remains the original checkpoint.
- Source identity is explicit: the successful classification study predates one
  exception-handling refinement allowing failed experimental tree refits to reuse
  an existing checkpoint. Reconstructing that single condition recovers the exact
  recorded study hash. Final forced-failure tests cover both linear and tree
  refit failures. Normal model code in that recorded study is unchanged. Runtime
  measurements used the final source hash throughout all three segments.

Raw verification artifacts are `default-equivalence.json`, `preservation.json`
and `integration-tests.xml` under `experiments/first-batch/`. No official checker,
clean-install, memory-limit, unpublished maximum-size or full confirmation claim
is made. No production default was changed; no commit or push was performed.

## Prioritized next development

1. Isolate and reduce routed parent-process import/configuration overhead before
   expanding model search. The runtime study exposes a real deadline-quality
   tradeoff on wider data: at 25 seconds routed recovers a weak baseline where
   auto sometimes finishes the stronger model. Test moving training-only imports
   into the worker, with paired timings and deterministic prediction/selection
   checks at generous budgets. Preserve completed-baseline checkpoint writes.
2. Investigate the eight indicator-free clustering losses, including the worst
   GMM basin failure, and add genuinely informative missingness mechanisms.
   Preserve full-covariance restarts. Only then test a predeclared combined
   training-only policy; never select using reference ARI or dataset names.
3. Retain two-holdout classification as an optional candidate. Investigate rare
   multiclass failures and weighting/selection interactions across broader
   mechanisms and real tables. Compare all objective scenarios and severe raw
   losses; do not impose an invented accuracy or BA veto.
4. Preserve regression candidates and gains. Extend compatibility checks to
   fresh regression families in a later selector-stability batch; do not tune
   regression models in this batch.
5. Obtain official rules, contract and checker before choosing production task
   champions. Freeze candidate/source/pool before a separately approved unopened
   confirmation campaign. This batch does not initiate that campaign.

Keep total selection time fixed when comparing a larger portfolio. Test budget
pressure and memory on larger permitted shapes when official limits are available.

All work stays uncommitted for approval. The starting local/remote checkpoint is
`0ad79a77844ef1bf5b15750a3e39a0ab4a7a06ed`; no push is authorized by this batch.

Proposed checkpoint commit, after user review:
`checkpoint: add controlled AutoML experiments and runtime diagnostics`.
