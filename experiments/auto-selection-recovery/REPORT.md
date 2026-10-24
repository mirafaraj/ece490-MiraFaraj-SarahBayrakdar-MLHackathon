# Selection-model recovery (review candidate)

Base and protected main: `b047cef0be579319e50f275616e420528aebec49`.
Branch: `codex/auto-selection-recovery`. No study branch was merged.

| Full unchanged input | Before wall s | After wall s | RMSE before -> after | R2 before -> after |
|---|---:|---:|---:|---:|
| Bike, 13,034 training / 4,345 test rows | 47.55 | 47.48 | 236.5072 -> 98.8703 | -0.1557 -> 0.7980 |
| Metro, 35,885 training / 12,319 test rows | 47.55 | 47.53 | 1984.6451 -> 1909.9455 | -0.0008 -> 0.0731 |

Both complete-process calls finished below 60 seconds and **missed the 45-second target**.
These diagnostic metrics are not verified official scoring metrics.

## Measured diagnosis

Instrumented unchanged-main CLI calls used exact evaluator `cases.json` inputs and
seed 42; hashes were checked. Selection finished before the deadline. Bike chose
Extra Trees at 13.30 seconds after CLI entry and began its full refit at 13.31.
Metro chose histogram boosting at 8.61; the existing Extra Trees fallback began
at 9.42 after histogram left its full-fit call without completion. The caught
exception text is unavailable. A matrix-cap rejection is supported by the code
and dimensions (the observed selection width was 81), rather than directly logged. Neither forest finished;
workers were stopped and emergency mean outputs published. No final-model prediction
completed. A model-fit start includes its existing preprocessing. Standalone final
preprocessing duration and time inside the interrupted estimator are unavailable;
there is no evidence that output serialization caused either failure. Original
profile/load timings: Bike .063/.140 seconds; Metro .078/.203 seconds. Final output
writing took about .05-.06 seconds. Child imports/startup are included in wall time.

## Exact repair

Retain the existing winning selection-partition fit, predict test rows with that
same fit, and publish through the existing validated atomic checkpoint machinery
before attempting the unchanged full-data refit. No extra fit, candidate, parameter,
preprocessing, selection score, objective, seed, cap, reserve, deadline or default
change. Existing routed and bounded execution paths remain intact. A successful
full-fit payload takes precedence; the emergency safety net remains available.

Recovered outputs explicitly report `fallback_kind=selection_model`,
`fitting_scope=selection_partition`, actual `training_rows=3000`, and available
training rows separately. Bike recovers a 40-tree ExtraTreesRegressor (fit 5.64s,
selection 6.86s, checkpoint prediction .14s); Metro recovers HistGradientBoostingRegressor
(fit .52s, selection 2.42s, checkpoint prediction .25s). Both profiles were independently
checked against full training tables (113/83 legacy fields plus 9/4 categorical
summaries). Reference labels were read only by the external scorer after output.

## Verification

- Full suite run once: **355 passed, 0 failed, 0 skipped**, including all 17 new
  selection-recovery tests. These cover actual fitting rows, corrupt/absent files,
  invalid metadata, pre-refit publication, no additional fit, publication failure,
  lexical classification labels, timeout emergency safety and full-result priority.
- All 120 saved CLI configurations matched prediction hashes, selected models,
  selection metrics and profiles exactly. Selection scores/skips/errors also matched.
  This includes all 39 explicit stress strategy combinations and all 36 default real
  cases, plus existing probes/duplicates. Zero emergency outputs and zero new
  selection-checkpoint recoveries among these previously successful runs.
- Stress maximum 26.60s; maximum across all 120 configurations 41.72s.
- Independent full-data profile checking passed; no profiling code changed.
- The 66 queued historical real baseline/routed comparisons not covered by that
  inventory were **not launched**, following the user's instruction to finish the
  running validation without further expansion. They remain a coverage gap; this
  is not a claim that all 108 historical real strategy combinations were rerun.
- Main remains b047cef and its tracked file hashes remain unchanged. No experimental
  study was integrated. No commit, merge or push occurred.

Before timings include local diagnostic event logging; their small differences
from after timings are not evidence of a speedup. Local source/input hashes,
per-case audits and test XML are preserved outside this worktree. Temporary newline
normalization restored original source formatting without changing executed code.

Changed files only: `src/unknown_dataset/models.py`, `src/unknown_dataset/runner.py`,
`src/unknown_dataset/supervised.py`, `tests/test_routed_checkpoint.py`,
`tests/test_selection_checkpoint.py`, and this report. Datasets, full prediction
and reference arrays, caches, environments and bulky logs are excluded.

## Limitations

Recovery preserves a smaller learned model, not a completed full-data fit. It
substantially helps Bike and modestly helps Metro on these inspected cases, with no
claim of broad quality generalization. The unchanged worker deadline still yields
about 47.5 seconds full-process latency, leaving less headroom than the target. No
performance or deadline tuning is included. Main remains the known-good rollback.

## Independent verification

The evaluator independently verified staged source/test hashes, the saved 355-test
result and all 120 saved prediction/profile comparisons. Fresh CLI runs reproduced
Bike RMSE 98.8703, R2 0.7980 in 47.50s and Metro RMSE 1909.9455, R2 0.0731 in
47.58s, with identical saved predictions and full-data profiles. All 46 focused
selection/routed recovery tests passed. No production changes were made during
independent verification. README and MODEL_CARD were subsequently refreshed for
submission; model code remains the independently verified candidate.
