# Targeted development batch

Starting checkpoint: `0ad79a77844ef1bf5b15750a3e39a0ab4a7a06ed`.
These are inspected development results, not unopened confirmation or estimates
of official points. The official scoring formula, checker and size limits remain
unavailable. No change in this batch is promoted to the default strategy.

The three investigations own separate production files. Timed studies run
serially, with no concurrent model experiments. The independent evaluator under
`reports/grade-evaluation/` and all historical reports remain unchanged.

Optional controls are partial router configuration overrides. They apply only
when explicitly selected with `--strategy routed --router-config PATH`:

```json
{
  "experimental": {
    "classification_weighting": "both",
    "classification_holdouts": 2,
    "classification_objective": "balanced_accuracy"
  }
}
```

Weighting accepts `objective` (incumbent default), `balanced`, `unweighted`, or
`both`. Objective accepts `route` (incumbent default), `accuracy`,
`balanced_accuracy`, or `macro_f1`. Holdouts accept 1 (default) or 2. The bounded
experimental selector shares one elapsed-time selection allowance across all
variants and folds; individual fits are still bounded by the outer worker deadline.
The original completed baseline checkpoint remains available.

Clustering alternatives are isolated overrides:
`clustering_missing_indicators: false`, `clustering_covariance: "diag"`, or
`clustering_kmeans_only: true`. Defaults remain indicators enabled, full covariance
and the existing BIC selection. Do not combine alternatives before their isolated
effects are reviewed. Clustering selection never accepts reference labels.

`--timings` enables CLI diagnostics; `experimental.runtime_instrumentation: true`
enables nested routed phase timings. Baseline and auto expose aggregate worker
times; routed exposes additional phases. Parent output serialization is reported
on stderr after writing outputs, so it does not require another output rewrite.

Detailed evidence and interpretations are in classification.md, clustering.md,
and runtime.md. The integration summary records the final test outcomes and the
next development priorities. Raw results are under `experiments/first-batch/`.

If an experiment proves unsuccessful, retain its optional configuration and
findings for review, and continue running the default configuration from the
checkpoint. A separate checkout at the checkpoint is the safest way to inspect
or run the earlier version without discarding these uncommitted findings. No
reset, clean, or history rewrite is needed.
