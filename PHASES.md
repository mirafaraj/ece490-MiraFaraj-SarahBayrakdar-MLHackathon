# Unknown Dataset AutoML — complete revised project plan

## Status and evidence

The user subsequently authorized resuming implementation. Router policies, completed-baseline
checkpoint recovery, the disabled-by-default categorical candidate, explicit objectives and
zero-default tie handling are now integrated. The current complete test suite passes 153 tests;
offline selector diagnostics passed 148 runs. Frozen CLI comparison passed 54/54 runs with stable
source/config and no fallbacks; routed matched auto quality but took longer. Eight strengthened
invariance tests also passed after the complete suite. Reports are in reports/router_validation.md.
Official alignment and promotion constraints below remain in force.

Previously verified: classification/regression/clustering MVP, training-only preprocessing,
structured profiling, lexical labels, missingness indicators, category-token isolation, staged
outputs, deadline worker, emergency fallback, pinned dependencies, and reproducible evaluation.
Existing supervised auto compares linear/tree/boosting candidates; clustering has an optional
restart improvement grounded in our own reproduced failure.

Historical evidence before router work: 68 passing tests; 48 valid broad CLI
runs over 12 families/two strategies/two seeds; and 20 valid fresh-seed clustering runs with zero
fallbacks. These historical results do not certify later versions, official compatibility,
maximum-size performance, clean installation, or remote CI. Preserve historical reports and hashes.

The supplied review inspected public EECE490 Stories Coffee code/reports; those results were not
independently reproduced. Their techniques are transferable hypotheses, not evidence of superiority
on our task. The linked pages could not be retrieved during this update, so no new source-specific
findings are claimed. Introduce no coffee-specific features or externally justified clustering change.

## Preserved constraints

- Six-hour contest, three students; one public GitHub repository is the sole submission, frozen
  at the deadline. Include runnable code, pinned dependencies, standard entry point, team names,
  README, and reproduction instructions. No hosted website is required.
- Eighteen private datasets: quality 85 points, equal contribution from the three tasks; profiling
  10; speed 5 conditional on useful performance. Sixty seconds per dataset on the same published
  CPU environment. Obtain exact metrics, timing boundary, size and memory limits from written rules.
- Metadata supplies task, feature types, and clustering k. Respect output order, labels, finite
  regression outputs, profile semantics, and required groups. Handle missingness, scale, categories,
  constants/duplicates, irrelevant features, imbalance, and nonlinearity within published limits.
- Development may use AI/open-source resources; submitted execution must be autonomous and CPU-only,
  without AI/network services. Preserve present thread limits pending verified environment changes.
- No organizer questions: use published kickoff files and shared corrections; log assumptions.
- **Baseline remains default.** All current objectives/compatibility claims remain provisional
  until official alignment and the promotion gates. Experimental candidates stay opt-in.
- Fit/select using training data only. Keep evaluation labels outside production APIs. Supervised
  test values are transform/predict inputs. Clustering's unlabeled input rows are its fitting data.
- Run expanded/multi-split validation offline; do not import that cost into assessment-time execution.
  Keep process deadlines, output reserve, fallback behavior, lexical labels, missingness, and contract
  guarantees. Missing official details block default promotion, not all provisional research.

## Phase 1 — Contract, objective, and foundation

**Implemented:** modular package, entry point, provisional CSV/JSON adapter, task dispatch, README,
local evaluation, dependency lock, and seeds. **Gap:** official contract/scoring/checker, environment
limits, and nine official practice datasets have not been supplied.

### 1A. Official metric alignment — highest priority

- **Dependencies:** published kickoff rules, scoring and input/output/profile specifications,
  checker and shared corrections; current adapters and baseline evidence.
- **Deliverables:** requirement-to-code/checker checklist; exact task metrics/direction/aggregation;
  labels-versus-probabilities requirements; timing/size/memory limits; profile semantics; resolved
  assumptions. Compare ordinary and balanced accuracy together and resolve weighting/decision rules
  against the official objective. Never infer the scoring metric from imbalance alone.
- **Acceptance:** fixture calculations agree with official metrics; checker/schema tests pass;
  all nine official practice datasets pass when available. Written sources support official claims.
  No change to default selection behavior before this gate; no official-points claim meanwhile.

## Phase 2 — Preserve MVP and complete optional routing safely

**Implemented:** Extra Trees supervised baseline; scaled K-means/MiniBatch K-means; existing auto
candidates; validated preparation/profiles/output handling; killable worker and weak emergency output.
**Implemented/tested:** explicit routing/policies/config, categorical opt-in, metric-aware supervised
selection, baseline reuse/checkpoint recovery, strict prediction checks and failure provenance.
Independent review and full-suite checks pass; task-policy promotion remains unapproved.

### 2A. Task policies and resources

- **Dependencies:** stable interfaces, existing models, Phase 1 objectives or explicit provisional
  settings; reports/ROUTER_PLAN.md and ROUTER_INTERFACES.md as architectural references.
- **Deliverables:** deterministic training-only summary, versioned RoutePlan, separate task policies,
  strict configuration, candidate exclusions, and run.json traces. Rules determine eligibility;
  training-only validation chooses supervised winners. Avoid dataset-name/seed rules or unvalidated
  correlation shortcuts. Bound candidate count, sample size, encoded width, working copies, estimator
  state, final refit, and prediction batches. Keep classification/regression/clustering objectives separate.
- **Acceptance:** baseline/auto parity, complete route/config tests, fixed-input deterministic rules,
  explicit metric/skips, and integrated routed tests. Resource observations do not certify whole-process
  memory compliance. New routed behavior stays opt-in until Phase 4 promotion.

### 2B. Completed-result recovery

- **Dependencies:** validated baseline output, parent deadline control, route traces, schema checks.
- **Deliverables:** atomic completed-baseline checkpoint before optional work; replace it only with
  complete validated predictions. Distinguish normal completion, completed-baseline recovery, weak
  majority/mean/round-robin emergency output, and invalid runs.
- **Acceptance:** forced optional timeout/error/crash returns valid completed baseline; corrupt
  checkpoints are rejected; no-checkpoint cases preserve documented behavior. Maintain staged outputs,
  stale-file cleanup, lexical labels, finite predictions, required k, and hard worker interruption.

## Phase 3 — Correctness, leakage invariance, and reproducibility

**Implemented evidence:** contract/profile/model tests, independent reviews/fixes, seeded preparation,
and forced deadline tests. **Now verified:** evaluation-label and supervised test-feature invariance,
row-permutation behavior and fitting-fold categorical statistics. **Gaps:** clean installation,
remote CI, and maximum-size/memory evidence.

### 3A. Leakage invariance

- **Dependencies:** explicit fit/selection boundaries, deterministic seeds, restored integration
  baseline, Phase 2 traces, and an isolated evaluation harness.
- **Deliverables:** mutate/permutate evaluation labels while holding all training/features fixed;
  compare prediction bytes and selection. Separately change supervised test feature values at fixed
  schema/count; compare fitted preprocessing, selected features, eligible models, validation scores,
  and selected model. Test row permutation must only permute predictions. Recheck fold-level fitting
  for every candidate, including new categorical statistics.
- **Acceptance:** evaluation-label changes cannot change predictions or selection; supervised test
  changes cannot alter training-fitted state or selection, though predictions may change. Reference
  labels cannot be passed into selection APIs. Use nonbinding budgets for semantic invariance and
  separate deadline tests to distinguish scheduling noise. For clustering, hidden-label invariance
  applies, but changing its unlabeled fitting rows may legitimately change groups. All existing
  contract, lexical-label, missingness, publication, and deadline tests continue to pass.

### 3B. Environment and resource verification

- **Dependencies:** integrated source/lock file; official environment and maximum sizes when supplied.
- **Deliverables:** clean-install log, complete test report, remote CI run, CPU/wall-time and memory
  stress evidence, assessment-environment reproduction commands.
- **Acceptance:** independent clean installation reproduces recorded behavior; CI executes; largest
  published cases satisfy external timing/memory/output limits. Bundled Python 3.12 success and a
  configured workflow alone do not satisfy these gates.

## Phase 4 — Offline improvement and selector validation

**Implemented:** offline real/synthetic families; per-case quality/time/validity/fallback records;
input/source hashes; historical snapshots; independent reviews. **Now implemented/measured:**
eight-family/two-split selector diagnostics, frequencies, retrospective gap, dummy comparisons,
categorical experiment and exploratory positive tie study. Default tie tolerance remains zero;
the categorical candidate remains disabled. **Gaps:** official objective alignment, larger independent
studies, and evidence/environment compliance adequate to promote new defaults.

### 4A. Selector stability before expanding the search

- **Dependencies:** verified current candidates, explicit objective (provisional or Phase 1 official),
  Phase 3A checks, frozen split protocol, and evaluation code inaccessible to selection.
- **Deliverables:** compare dummy, linear/tree/boosting candidates, auto, and routed on identical
  outer splits over fresh datasets and multiple split seeds. Use inner training-only selection.
  Report selection frequency, baseline wins/ties/losses and magnitudes, per-family failures,
  validation-to-outer-score gaps, and selected-versus-best-evaluated candidate gap. Also compare
  routed eligibility against the same unrestricted portfolio to isolate routing's contribution.
- **Acceptance:** reproducible per-split distributions; no reference-label access; empirical regret
  is best evaluated score minus selected score for higher-is-better metrics, reversed for losses.
  The best candidate is a retrospective diagnostic, never an oracle available to production. Mark
  unevaluated candidates/comparisons incomplete rather than zero regret. Report uncertainty by
  dataset; overlapping splits are not independent new datasets. Include losses, not only gains.

### 4B. Inexpensive categorical classifier — experimental

- **Dependencies:** Phase 4A harness/metrics, Phase 3A fold isolation, eligible categorical columns,
  and a bounded development-defined feature/smoothing search.
- **Deliverables:** a single-feature classifier with fitting-fold class prior pi_k and counts:
  `p(k|c) = (n(c,k) + alpha*pi_k) / (n(c) + alpha)`, alpha > 0. Unseen categories use pi_k.
  Treat missing categories separately from genuine values and preserve lexical identity. Fit all
  counts/priors inside each fitting fold. Select feature and any small alpha grid using inner
  training validation only, then refit on all training rows. No global target encoding or test-label
  counts. Bound cardinality, memory, feature search, and prediction-time candidate count.
- **Acceptance:** hand-calculated smoothing/prior tests; normalized finite probabilities where
  exposed; unseen/missing/single-class cases; explained no-category skip; fold and leakage-invariance
  tests. Compare with dummy/prior, linear and tree candidates on identical outer rows, including
  runtime, validity and fallbacks. Preserve opt-in status unless fresh confirmation supports promotion.
  Inexpensive fitting alone does not justify adding it to every assessment-time portfolio.

### 4C. Quality/runtime tradeoffs and near-tie rules

- **Dependencies:** Phase 4A quality/timing distributions, objective direction/scale, resource
  policies, and explicit candidate-complexity definitions.
- **Deliverables:** joint quality, full wall time, validity, and fallback tables/Pareto comparisons.
  Investigate preferring faster or simpler candidates for effectively tied validation scores.
  Define tolerance, metric scale, simplicity ordering and timing method on development data;
  freeze them before confirmation. Use fitting-split timings or development-calibrated cost estimates
  in production, never outer-test quality or confirmation timing to choose that same case's model.
  Report selection/refit/prediction/full CLI cost separately from offline evaluation cost.
- **Acceptance:** frozen tie tests select consistently; materially better scores are not discarded
  as ties; compute savings and quality losses are reported against the no-tie rule. No invented
  universal tolerance or speed-only success. Include failures and distinguish completed-baseline
  recovery from weak emergency fallback. Freeze promotion-loss tolerances before confirmation too.

### 4D. Frozen confirmation, independent review, and promotion

- **Dependencies:** frozen source/config/candidates/tie policy; relevant Phase 3 checks; predeclared
  objective and acceptance tolerances; genuinely fresh families/datasets and uninspected splits.
- **Deliverables:** controlled ablations of baseline/current auto/routed/categorical/tie variants;
  identical input hashes and source/config stability; per-task/family quality, regression magnitude,
  wall time, memory, validity and fallback categories. All expanded/repeated validation runs offline.
  Treat every previously inspected confirmation result as development for later changes, including
  observed seeds 42, 314, 2027 and 4099. Any subsequently inspected seed, including 8081, is also
  development thereafter; an uninspected designation is only prospective confirmation. Predeclare
  fresh splits/families before evaluation; do not tune after inspecting confirmation results.
- **Acceptance:** complete paired reports and independent review, no label leakage, no omitted
  invalid runs, and no post-confirmation threshold adjustment. Default promotion additionally
  requires official alignment (1A) and verified environment limits. Promote task policies separately.
  Missing official objective, inconclusive evidence or unacceptable regressions preserves baseline.
  Clustering changes require our own failure analysis; external business-project examples supply
  no clustering-performance evidence.

## Phase 5 — Model card and submission

### 5A. Concise model card

- **Dependencies:** Phase 1 contract/assumptions, verified implementation status, measured Phase 3/4
  reports, and documented selection/fallback semantics. A marked-provisional draft can start earlier.
- **Deliverables:** MODEL_CARD.md describing supported tasks/inputs/sizes; preprocessing/labels;
  candidates, metrics, fitting/selection boundaries; seeds/dependencies/source/config hashes and
  commands; runtime environment/evidence; checkpoint versus emergency fallback; limitations.
  Clearly separate measured results, assumptions, partial work and unverified official compatibility.
- **Acceptance:** quantitative claims link to measured report/source/config; ordinary-versus-balanced
  accuracy tradeoff is explicit; no unsupported official, hidden-test, maximum-size, clean-install or
  CI claims. Keep it concise and consistent with README, which remains the setup/contract reference.

### 5B. Official submission readiness

- **Dependencies:** official contract/checker/practice data; chosen tested source; clean install/CI;
  team metadata; README/model card; assumption log.
- **Deliverables:** public GitHub URL, runnable pinned project, entry point, team names, reproduction
  commands, official-checker results, assessment-environment evidence and final submitted commit.
- **Acceptance:** independent reproduction of all nine official practice tasks/profiles; public
  repository access; required deadline commit frozen by hour six. No website or external inference
  dependency is introduced.

## Execution order when implementation is authorized again

1. Verify interrupted edits against the last known-good source; do not call them completed.
2. Obtain written rules (1A); provisional offline design/evaluation can proceed while awaiting them.
3. Complete router/checkpoints (2), then leakage/integration checks (3A), before new selector work.
4. Measure existing-selector stability (4A); develop the categorical experiment (4B) and tie policy
   (4C); freeze settings before confirmation (4D).
5. Complete independent review/environment verification, update model card, and submit (5).

Three-student ownership: contract/integration/runtime; modelling/policies; independent evaluation,
leakage and reproducibility. Delegate disjoint files only after interface agreement. Under contest
time pressure, valid baseline, official alignment and submission checks outrank optional experiments.
The later implementation request authorizes the current run; it does not waive external gates
or authorize default promotion or external publication.

## References and evidence

Examples supplied from public-code/report review; not independently reproduced comparisons:
- https://github.com/marwamansour256-lab/EECE490-Hackathon/blob/main/EVALUATION.md
- https://github.com/rakanhmd/eece490-hackathon/blob/main/src/forecast.py
- https://github.com/sarmadosman/eece490-hackathon-2026/blob/main/reports/forecast_evaluation.md
- https://github.com/mouhammed-soumaille/EECE-490-Hackathon/blob/main/src/storiespair/models.py

Preserve reports/RESULTS.md, reports/clustering_improvement.md, independent review reports,
experiments/baseline.json, broad comparisons/summaries, clustering experiment reports, source
snapshot and requirements.lock as historical evidence. reports/ROUTER_PLAN.md is the architecture
supplement; this document is the authoritative phase/dependency/acceptance/promotion plan.
