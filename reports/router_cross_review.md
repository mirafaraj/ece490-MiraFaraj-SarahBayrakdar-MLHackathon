# Independent router integration review

Review completed during the implementation snapshot, before the orchestrator's
final source freeze and confirmation evaluation. This agent owns the runtime
implementation; this cross-review independently inspected the facade, policies,
supervised portfolio, and categorical candidate implemented by other agents.
No production source was changed during this review.

## Outcome

No release-blocking correctness or leakage issue found in the inspected snapshot.
Keep `routed` opt-in and the categorical candidate experimental until independent
confirmation results and official objectives are available. This review does not
establish model superiority, a whole-process memory ceiling, or contest compliance.

## Verified paths

- `models.fit_predict` defaults to baseline and dispatches baseline/auto independently
  of the new routed arguments. `supervised.fit_predict_supervised` executes its
  previous auto branch when `route_plan` is absent. Existing model/candidate checks pass.
- Routing eligibility inspects supplied training features/labels and explicit
  configuration. No evaluation references or dataset-name rules are present in the
  inspected policy paths. The full-training summary is used for eligibility, not
  as a fitted transformer for inner validation.
- Supervised model fitting, imputers/encoders, and categorical counts are fitted on
  inner fitting rows. Each candidate uses the same inner validation rows. Final
  refitting uses all supplied training rows only after candidate selection.
- Classification objectives control class weights; regression losses are minimized,
  classification scores maximized. The fractional-label metric bug previously
  identified by the orchestrator is fixed using temporary joint label encoding.
- Near-tie selection compares each score with the global best rather than allowing
  pairwise tolerance drift. Categorical feature/alpha selection is restricted to
  fitting-data eligible features and bounded configured choices.
- The categorical classifier separates missing tokens from lexical values, preserves
  leading-zero class/category strings, returns the fitting prior for unseen values,
  and adjusts balanced decisions separately from its reported posterior.
- The routed facade completes full baseline prediction before invoking the checkpoint
  callback. Checkpoint publication is synchronous in the runtime. Optional model
  exceptions preserve baseline predictions with an explicit reason. Hard worker
  termination recovers the separately validated checkpoint through the runtime.
- Test-row values influence final predictions but do not enter routing summaries,
  holdout fitting, or validation scores. A test-feature mutation invariance test passes.

## Nonblocking findings and limits

1. Optional supervised selection may perform another full Extra Trees refit after
   already completing and checkpointing that same task baseline. An optional refit
   failure can also trigger a new baseline refit instead of returning the existing
   one. Reusing the completed baseline would save runtime. Current hard process
   termination and checkpoint recovery preserve useful output if this exceeds budget.
2. Candidate eligibility and selection reserve time, but optional final-fit duration
   is not predicted from measured sample-fit costs. The fixed reserve is a heuristic,
   not a guarantee of final-refit completion. Keep complete CLI timing and checkpoint
   recovery rates in the benchmark; large-input stress testing remains necessary.
3. Working-memory estimates and dense-cell caps are candidate eligibility controls,
   not OS-enforced memory limits. Baseline fitting is always eligible. Prediction
   batching uses the dense-cell cap; estimator state, Python dictionaries, sparse
   intermediates, and native allocations can exceed a user-supplied memory estimate.
4. Multiple categorical feature/alpha combinations use one inner validation split.
   That split is model-selection evidence, not independent evaluation. Separate
   confirmation families/seeds must remain untouched until implementation freeze.
5. The baseline callback receives the facade's metadata dict synchronously. The
   facade later updates that dict's stage/reason when returning baseline-only/error
   output. Runtime JSON serialization snapshots it immediately, so persisted
   checkpoints are safe; custom callbacks storing references should copy metadata.

## Verification

Command:

```powershell
.venv312/Scripts/python.exe -m pytest tests/test_routed_supervised.py tests/test_categorical.py tests/test_routing.py tests/test_models.py tests/test_supervised_candidates.py -q --basetemp .pytest_tmp_router_cross_review
```

Result: **50 passed in 8.31 seconds**. One expected K-means convergence warning on
identical rows; the required nonempty cluster occupancy repair test passes.

Separate runtime implementation verification before this review passed 32 checkpoint,
spawned-worker, existing CLI, and existing deadline checks, followed by 20 unit checks
after opaque-label validation hardening. The orchestrator still owns the final full
suite and source/config freeze checks.

Inspected source SHA-256 values:

| File | SHA-256 |
|---|---|
| routed.py | `8B19B80FB9355A5315122570F89297953B76CD948141B2F86908CADA91F51DDB` |
| policies.py | `AAFE829099D87B5DEC31CED7069B181EC9AC2C9511797543EB0EF14EA92A29A1` |
| supervised.py | `D6E32ACE322DEA17E147E22DFD802301C469BEC59C7FE06B607C8C98BC0E1C7A` |
| categorical.py | `762A29482208E3FBFAFACB32C10AB9020C4BCAC642ED90BB7BC86E84200A1B2B` |
| routing.py | `D4C03AB5FA29F46E0A49469CFD6AB1EAF50C96F3DB7F605419DD604DFF7B1343` |
