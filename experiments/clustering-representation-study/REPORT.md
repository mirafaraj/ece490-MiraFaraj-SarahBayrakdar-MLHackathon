# Minimal clustering study and separate default change

Completed 2026-10-23 on `codex/clustering-representation-study`, starting at `36e0bf421eb6dc8615e781aa0709a8017c073777`. Main and backup remain unchanged. No commits, pushes, downloads, dependency changes, security changes or fresh confirmation runs.

## Separate authorized default change

CLI and public `fit_predict` now default to **submission**: classification -> baseline; regression/clustering -> auto. All three original strategies remain explicit options. No estimator, shared preprocessing, routing policy, profile or recovery implementation changed.

On the existing wine/diabetes/iris seed-42 inputs, before/after verification ran **24 actual CLI invocations and 24 API calls**. All nine explicit baseline/auto/routed comparisons retained identical prediction hashes, selected models and CLI profiles. For every task, omitted strategy matched explicit submission and its delegated strategy in both API and CLI. These are three representative inputs, not exhaustive prediction equivalence over all possible tables. Existing explicit paths are otherwise unchanged in source.

The mapping reflects development judgment under unknown official metrics. Classification baseline retains the documented accuracy/macro-F1 versus balanced-accuracy/minority-recall tradeoff: previous auto comparison gained 3.90 balanced-accuracy and 11.78 minority-recall points but lost 1.57 accuracy and 1.45 macro-F1 points. These are diagnostic results, not official points.

## What ran and what it answered

Information checks passed before the comparison. A numeric column with one observed value plus a varying mask survives constant removal. Different masks survive duplicate handling; exact duplicate columns retain one representative. Imputation retains the indicators. Categorical missingness remains distinguishable in the focused low-cardinality test. **No demonstrated defect justified a model change.** This does not certify all possible categorical cardinalities.

Frozen protocol: 11 representative existing conditions, 12 new synthetic conditions (two mechanisms x balanced/90-9-1% sizes x three seeds), and two Mall conditions. **75 main fits**, **30 Mall subsample fits**, and six subsequent exact-repeat checks. All use existing estimators/controls, fixed matching inputs/k/seeds, without parameter search. Numeric indicators are full strength, .25, or omitted. All new data, references and prediction arrays are under `local/` and must remain outside any proposed commit.

### Recovery quality: ARI / AMI

| Condition | Full | Quarter | Omitted |
|---|---:|---:|---:|
| Random/intrinsic-random masks | .3617 / .3590 | .4224 / .4233 | .2681 / .3073 |
| Informative masks | .2039 / .2590 | .1234 / .1405 | .1096 / .1329 |
| Missingness-only, balanced | 1.0000 / 1.0000 | 1.0000 / 1.0000 | approximately 0 / 0 |
| Missingness-only, rare | .7825 / .7629 | .6680 / .6819 | -.0059 / -.0045 |
| Unrelated block masks, balanced | -.0011 / -.0013 | .6557 / .6482 | .9818 / .9684 |
| Unrelated block masks, rare | -.0024 / -.0032 | .1377 / .2694 | .3469 / .4432 |

Seeds are averaged within conditions and families first. Overall equal-family ARI/AMI means: full .52107/.51267; quarter .57131/.56958; omitted .43205/.44129. These deliberately selected development probes do not estimate hackathon points. The nuisance mechanism is intentionally a competing grouping signal, not a new real family.

**Do not hide the losses:** quarter's worst ARI change is **-.34347** on missingness-only/rare seed104743 (.75484 -> .41137); informative mixed losses include -.10711 and -.21520. Omitted loses approximately **1.0 ARI** on every balanced missingness-only seed and fails all rare missingness-only seeds (smallest-group recall zero). Full/quarter recover every smallest rare group member in those mask-only cases, but their overall partitions still differ. Every reference case is in `PER_CASE.md`; `per-case-metrics.csv` also records AMI changes, group sizes, repairs, selected models, validity and direct runtime. Nothing is dropped from the table.

Mechanism: indicators preserve useful mask-only structure, but can overweight unrelated site/batch patterns. Reducing them can help the latter and harm the former. In the worst quarter rare case both policies choose full GMM; uniformly scaling the nonconstant indicator dimensions still changes initialization/regularization behavior. This experiment does not isolate those GMM effects further. Exact repeats of all three representations on that case and a nuisance case retained identical predictions. Convergence/BIC/stability alone does not establish intended groups.

**Decision: retain incumbent full-strength clustering.** Quarter fails informative and rare-group gates despite a positive aggregate mean; omission fails absolute mask-only recovery and increases empty-group repairs. No candidate qualifies for fresh confirmation. Proposed external confirmation families remain unacquired and unopened. No selector was introduced.

## Mall: unlabeled, k=5 assumed

One original copy, CustomerID excluded, all four remaining features retained. No ground-truth ARI/AMI is reported. All fits/stability subsamples produce five occupied groups, without repairs.

Clean Mall gives identical predictions across all representations: common-space silhouette .3074, mean subsample partition stability .6728. With fixed random-30% numeric missingness, silhouette is full .2347, quarter .1449, omitted .1568; stability is .6125/.7033/.5789. Stability compares fitted partitions on shared rows, not truth labels. Five fixed 80% subsamples per representation/condition. Common-space metrics use the incumbent representation; they are descriptive and do not establish which grouping is semantically correct. Own-space internal metrics are not used to choose a winner.

## Reliability, verification and limitations

All 75 main and 30 stability fits valid; no emergency or other fallback. Main-fit empty-group repairs: full 1, quarter 1, omitted 12 (two per mask-only case, required after removing all variable information). Direct per-case maxima: .359/.171/.203 seconds; these exclude interpreter startup. The full direct study took 10.641 seconds after imports, including stability and reporting work.

Seven additional actual CLI checks used 50-second budgets/60-second outer limits: all representations on the rare mask-only case and a nuisance case, plus Mall/submission. All outputs valid, no fallback/recovery, predictions exactly match corresponding direct outputs. Maximum **10.188 seconds**, with none above45 or60 seconds. These are small-case checks, not large/wide or organizer-hardware certification. Per-case peak memory was unavailable and is not inferred from timings. Existing large-input safeguards were preserved and covered by correctness tests, not re-benchmarked in this minimal study.

All 25 full-data profiles independently reconstructed using the existing separate reference implementation; discrete fields exact, floats rtol1e-10/atol1e-12. References remain outside model inputs and the five focused tests include reference/test-feature invariance for all representations. No raw datasets were downloaded. Existing families/masks/seeds remain development evidence; two new mechanisms and Mall do not establish broad generalization.

Final complete existing suite: **295 passed**, three expected warnings. Focused information suite: **5 passed**. Initial suite evidence (294 passed/one failure) is retained locally: a routed instrumentation test required nonempty optional scores at25 seconds, but cold startup left a16.906-second worker budget below the unchanged17-second selection gate. Only this test's budget/timeout became50/60 seconds, preserving every selection/prediction/timing assertion. No production deadline was relaxed. Final suite rerun passed.

## Exact source/document/test changes

- `src/unknown_dataset/models.py`: public default and API docstring only.
- `src/unknown_dataset/runner.py`: CLI default only.
- `README.md`, `MODEL_CARD.md`: current mapping, API example, assumptions/tradeoff and historical-default clarification.
- `tests/test_reliability_preparation.py`: submission-default assertions and explicit baseline resource tests.
- `tests/test_clustering_candidates.py`: baseline comparators explicitly request baseline.
- `tests/test_batch_runtime.py`: generous-budget instrumentation check as described above.

New reusable study code: `verify_defaults.py`, `test_information.py`, `compare.py`, `verify_study.py`, `summarize.py`. New concise evidence: `defaults-before.json`, `defaults-after.json`, `candidate-manifest.json`, `candidate-manifest-verified.json`, `protocol.json`, `summary.json`, `verification.json`, `PER_CASE.md`, `per-case-metrics.csv`, `mall-metrics.csv`, this report and updated `PROGRESS.md`. Stage1 proposal and supplied-file audit already existed and remain unchanged. `local/` contains excluded generated inputs, references, arrays, CLI outputs and test artifacts; preserve it locally.

The original reliability freeze record is untouched. The first candidate manifest records the isolated default phase; the verified candidate manifest records final hashes, dependencies, exact changed/new files, verification artifacts and local input/reference hashes without embedding rows or arrays. Working changes remain uncommitted on the study branch. This is development, not final submission or a clustering promotion. Exact Saturday deadline remains unconfirmed.
