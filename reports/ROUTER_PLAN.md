# Task-specific policy router: proposal and implementation plan

The authoritative revised phases, dependencies, deliverables, acceptance criteria and promotion
rules are now in [PHASES.md](../PHASES.md). They incorporate official metric alignment, selector
stability, the experimental categorical-count candidate, frozen near-tie handling, leakage
invariance and the model card. This file remains an architecture supplement; its earlier phase
outline is not a second work queue. After the later implementation authorization, router integration
passed 153 tests, 148 offline diagnostic runs and 54 frozen CLI comparisons; see
router_validation.md for final evidence, costs and observed regressions. Baseline remains default and official compatibility
remains unverified.

## Assessment

Recommended: evolve the existing task dispatch into explicit task-specific policies. Current
models.py already routes classification/regression/clustering and baseline/auto; supervised.py
and clustering.py have eligibility rules. The improvement is centralizing policy, objective,
resource limits, traceability, and preservation of completed predictions, rather than adding
another task switch. This proposal does not implement or promote a new router.

Rules should decide which candidates are eligible. Supervised validation decides between eligible
candidates. Clustering uses bounded unsupervised diagnostics and conservative fallbacks because
its private reference-group objective is not observable. No rule should reference dataset names,
benchmark seeds, hidden labels, or a guessed task. Metadata supplies the task and required k.

## Inputs and decisions

Training-only routing summary: row/column counts, declared feature types, estimated encoded width,
numeric/categorical proportions, missingness, constant/duplicate counts, class cardinality and
counts for classification, plus resource limits and an explicitly configured scoring objective.
Regression label shape/skew may be diagnostic; it must not trigger an irreversible target transform.
Preprocessing fitted during model selection remains confined to each fitting split. Test labels
are never available. Test row count may inform prediction-cost reservation; test values must not
drive model selection or fitted preprocessing.

Output RoutePlan: policy/rules version, eligible candidates with their preprocessing configuration,
split strategy, selection metric, sample/candidate limits, deadlines, reserve for final fit/prediction,
and explanations for skipped candidates. Eligibility and routing must be deterministic for the
same metadata/summary/configuration. Elapsed-time cutoff behavior is necessarily resource dependent.

## Initial task policies

| Task | Candidate rules | Selection and safeguards |
|---|---|---|
| Classification | Small bounded Extra Trees baseline; add logistic regression and histogram boosting when shape/memory/budget permit | Stratified training-only validation where feasible; preserve opaque labels; rare/single-class bypass; weights/thresholds follow the configured metric rather than imbalance alone |
| Regression | Extra Trees baseline; add scaled Ridge and histogram boosting when eligible | Training-only RMSE or explicit official objective; refit selected pipeline; target transforms are later experiments, evaluated on original target scale |
| Clustering | Scaled K-means or MiniBatch K-means; mixture candidates restricted to eligible numeric shapes | Honor metadata k; preserve numeric/category weighting policy; deterministic restart fix; bounded BIC/stability diagnostics; no claim that an internal score guarantees private ARI |

Wide/sparse/high-cardinality inputs prune dense candidates before allocation. Dense matrix limits
do not equal a whole-process memory limit; estimate copies, encoders, estimator state, and prediction
batches too. Large inputs reduce candidate count/sample size. Low budget returns the best completed
valid result, with the original emergency fallback only when none exists. Do not infer linearity
from a correlation threshold and hard-route to a single model.

## Execution and checkpointing

Preserve the current external deadline and output reserve. Produce and validate baseline
predictions first, then atomically checkpoint them in the worker workspace. Optional model work
must not overwrite that checkpoint until a complete valid replacement is ready. A candidate is
promoted only after validation selection and its final full-training refit/prediction complete.
If the worker is terminated, the parent uses the best completed checkpoint. Mark checkpoint-based
fallback separately from the existing majority/mean/round-robin emergency fallback.

Use a calibrated execution budget for loading/profiling, baseline, candidate validation, final
refit, prediction, serialization, and termination. Limit the initial portfolio to two or three
candidates. Deadline checks between estimator fits cannot interrupt a fit; keep killable workers.
Candidate selection should account for model size/refit cost, not only the sampled-fit time.

## Proposed modules

- routing.py: pure summary-to-RoutePlan logic and shared dataclasses.
- policies.py: versioned classification/regression/clustering rules and objective configuration.
- models.py: compatible facade delegating plan execution; preserve existing baseline and auto.
- supervised.py and clustering.py: task-specific fitting/selection implementations.
- runner.py: resource/deadline control and completed-result checkpoint recovery.
- configs/router.json: explicit limits/objectives; no executable configuration or dataset-name rules.
- run.json additions: policy version, observed inputs, rule decisions, exclusions, validation
  scores, selected estimator, actual timing, and fallback/checkpoint provenance.

Expose a new opt-in --strategy routed. Existing command defaults and contract remain stable until
the promotion gate passes. Optional task-specific metric configuration stays explicitly provisional
until the official scoring specification is supplied.

## Implementation phases and gates

1. Extract policies without changing predictions. Add summary/plan types, separate metrics,
   and route traces. Run existing tests and parity checks against baseline/auto. No new models yet.
2. Add resource eligibility and baseline checkpoint recovery. Test that dense candidates are
   pruned, constants/rare classes select valid routes, and a forced optional-worker timeout returns
   completed baseline predictions instead of weak emergency output. Retain output schema checks.
3. Add task-specific routed portfolios using existing models. Validate preprocessing boundaries,
   metric-aware weighting, budget/refit reservation, group count, and explanation metadata.
4. Freeze rules and compare baseline/current-auto/routed on identical inputs. Treat all already
   observed seeds (42,314,2027,4099) as development/regression evidence. Add genuinely new task
   families/real datasets and predeclare fresh confirmation seeds. Keep outer evaluation separate
   from each router's inner model-selection split. Do not tune after confirmation inspection.
5. Independent review and promotion. Check per-task/per-family quality, worst regressions, invalid
   outputs, fallback rates, full CLI time, and memory. Compare routing against the same portfolio
   without eligibility rules to identify whether routing itself helps. Set promotion tolerances
   before the experiment and in the official metric once known. Promote task policies independently;
   a useful regression policy does not imply clustering policy is safe to promote.

## Delegation after implementation authorization

Orchestrator: interfaces, routing facade, checkpoints, integration, and final comparison.
Supervised agent: classification/regression policies and split/metric correctness.
Clustering agent: clustering eligibility/diagnostics and limits.
Independent evaluation agent: fresh benchmark families, routing/timeout tests, and review.
Assign disjoint files and freeze interfaces before parallel edits.

## Limits

A router can improve resource use and selection discipline; it cannot reliably identify the best
algorithm for every unfamiliar table. The dataset suite is currently too small for a learned
meta-router or a large hand-tuned rule collection. Official schemas, objectives, size/memory limits,
and checker remain absent. Clean-install/assessment-environment verification is still required.

Technical references: scikit-learn cross-validation guidance
(https://scikit-learn.org/stable/modules/cross_validation.html) and clustering evaluation guidance
(https://scikit-learn.org/stable/modules/clustering.html). Internal clustering scores and reference-label
scores measure different properties; model-selection scores must not be reported as independent tests.
