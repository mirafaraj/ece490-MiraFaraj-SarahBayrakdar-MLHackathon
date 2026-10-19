# Router implementation interfaces and ownership

Implement a new opt-in `routed` strategy; baseline/auto semantics and default stay unchanged.

## Ownership
- Policy agent: routing.py, policies.py, configs/router.json, supervised.py, tests/test_routing.py,
  tests/test_routed_supervised.py. Preserve old supervised API behavior without route_plan.
- Runtime agent: runner.py, checkpoints.py, tests/test_routed_checkpoint.py. Parent owns models.py
  and routed.py facade; coordinate new optional arguments below.
- Evaluation agent: scripts/evaluate.py, scripts/datasets.py, scripts/summarize_comparison.py,
  tests/test_routed_evaluation.py, reports/router_validation.md. No other tests edits.
- Parent: models.py, routed.py, clustering.py optional route-plan gate, documentation, integration.

## Public facade additions
`models.fit_predict(..., strategy='baseline', route_config=None, checkpoint_callback=None)`
routes `routed` to `routed.fit_predict_routed` with the same kwargs, except strategy.
`route_config` is a loaded dict (None means repository default), never a filename.
`checkpoint_callback(predictions, model_info)` publishes a completed baseline result. It is
called by routed execution only. Callbacks receive JSON-safe model_info and 1D predictions.

## Policy APIs
`routing.summarize_training(task, X_train, y_train=None, feature_types=None)` -> RoutingSummary.
`routing.load_router_config(path=None)` -> validated dict. Defaults from configs/router.json.
`routing.make_route_plan(summary, time_budget=50.0, n_clusters=None, config=None)` -> RoutePlan.
RoutePlan has `to_dict()` returning JSON-safe dict, and attributes:
task, policy_version, metric, candidates (list[str]), excluded (dict[str,str]),
validation_mode, selection_sample_rows, selection_fraction, final_tree_count, dense_cells_limit.
Additional attributes allowed. Supervised candidates: extra_trees, linear, histogram.
Clustering candidates: kmeans, gmm. Always include the cheap task baseline.
Unknown config keys/objectives rejected, finite positive limits validated. Task policies should
use summary, dimensions, explicit objective, and budget, never dataset names/labels at evaluation.
Default provisional metrics: classification balanced_accuracy, regression rmse, clustering bic.
Classification accuracy and regression mae should also be supported explicitly.
Routing summary may inspect training labels for class counts; never test rows/labels.
`supervised.fit_predict_supervised(..., route_plan=None)` respects plan metric, candidate eligibility,
sampling/validation, dense cap, tree count; auto without plan retains current behavior.
Return model_info includes route plan and trace/skips where practical; no callback inside supervised.

## Parent routed executor
Fit baseline before optional work; callback it; compute remaining budget; execute supervised
plan or gated existing clustering geometry. Preserve completed baseline on optional exception.
Supervised baseline can use parent helper supervised._fit with configured metric if needed, but
prefer explicit `fit_baseline_supervised(..., route_plan)` API from policy agent that returns same
tuple and uses matching weights/metric. To avoid duplicate full fits on baseline-only plans, parent
simply returns checkpointed baseline. If a final fitted optional result exists, return it normally;
parent callback need only checkpoint the baseline. Decisions go into model_info['routing'].

## Runtime/checkpoint API
runner accepts --strategy routed and --router-config FILE (optional JSON config override).
Load config before worker; pass route_config only for routed strategy, retain existing CLI behavior.
Checkpoint storage is separate from final result file; complete writes published atomically.
Validate every recovered prediction's shape/count/class membership/finite numeric/k occupancy.
On routed worker timeout/error/crash, prefer valid checkpoint over emergency fallback. Mark
fallback=True, fallback_kind='completed_baseline', checkpoint_recovered=True, original cause.
No checkpoint -> existing emergency behavior for timeouts; unexpected failures still nonzero.
Keep actual process timeout and stale-artifact cleanup. Do not accept user-provided pickle inputs.
Parent exposes reusable prediction-validation helper only if convenient; runtime may own it locally.

## Evaluation
CLI evaluator accepts strategies baseline/auto/routed. Fresh suite: --suite router; include additional
offline real family (iris/digits/linnerud as appropriate) plus distinct synthetic families.
Previously seen seeds42/314/2027/4099 are development. Predeclare new confirmation seed8081.
Parent runs comparisons; evaluation agent can run development smoke and its tests, but must not
inspect confirmation references before code freeze. Preserve prior reports. Emit partial progress
per run; source/config fingerprinting should include router config. Benchmark checkpoint recovery
separately from weak emergency fallback in summary. Full CLI time is an observation, not guarantee.
