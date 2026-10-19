# P0 evaluator implementation

This change is confined to the independent evaluator. No model, training-time
selection, production entry point, dependency lock, or original experiment was
changed. No model experiments were run. Saved artifacts were read and new reports
were written under `p0-review/`.

## Policy

- No accuracy or balanced-accuracy veto, primary-metric default, or secondary-loss
  budget affects promotion. The former arguments remain deprecated diagnostic
  metadata. `--enforce-accuracy-gate` explicitly requests the historical diagnostic;
  its result is separate from optimization and promotion.
- Classification, regression, and clustering are assessed separately. Neither a
  classification loss nor an invalid classification run erases regression evidence.
- Provisional scenarios are accuracy/balanced accuracy/macro F1; RMSE/MAE; ARI/AMI.
  They are hypotheses about possible objectives, not official grading metrics.
  Directions and supplied official objectives can be declared with
  `--objective-policy`; missing official metrics are never replaced by proxies.
- Each scenario ranks candidates within the frozen comparison pool on identical
  inputs. Ties use average fractional rank; seeds/datasets are averaged within
  declared family, then families receive equal weight. Worst scenario mean rank
  regret expresses robustness to objective uncertainty. It is not official points
  or an estimate of expected total grade. It ignores magnitude, so raw worst cases
  and alerts must be reviewed too.
- Deterministic 10,000-resample bootstrap intervals use family groups, not rows or
  seeds. Single-family evidence receives no interval; related generator mechanisms
  can still invalidate an independence interpretation. Fewer than eight families
  or missing scoring scenarios prevent provisional advancement. Eight is a
  predeclared engineering threshold, not statistical proof of generalization.
- A supported positive robust difference can advance a candidate provisionally.
  A supported negative difference can reject a quality claim; faster execution or
  a different recovery path retains it for tradeoff review instead. Equal quality
  and consistently slower observed times are explicitly identified.
- Alerts trigger at >=10 percentage points for accuracy/balanced accuracy/macro
  F1/class recall, >=0.10 ARI/AMI, or >=25% relative RMSE/MAE increase. These are
  investigation thresholds, not permitted-loss budgets or automatic vetoes. All
  smaller losses remain in per-case evidence and worst-case lists.
- Invalid candidate outputs and actual external deadline failures block that
  task. Unknown hashes, incomplete inventories and nonfinite saved metrics cannot
  support a promotion. Completed-baseline recovery is distinguished from emergency
  predictions; valid recovered output requires quality review rather than being
  mislabeled invalid.

Official rules remain unavailable. Actual promotion requires a declared written
official source/hash, the measured official objective, a matching registered
incumbent, a predeclared frozen case/pool/policy manifest, fresh unopened cases,
adequate family evidence, and explicit external attestations for checker, profile,
clean environment, maximum size, memory, fallback quality and immutable-baseline
comparison. Severe-regression review is required when alerts exist. The evaluator
records these declarations; it cannot prove that a human never opened results or
that an external checker was honestly attested. It never silently promotes a
registry entry or estimates unavailable speed/profile scoring formulas.

## Interfaces and reproducibility

`analyze.py REPORT --candidate-strategy routed --baseline-strategy baseline
--registry REGISTRY --objective-policy POLICY --manifest MANIFEST --output NEW`
reads saved artifacts only. `--enforce-evidence` returns nonzero for invalid or
incomplete evidence; a quality tradeoff alone does not fail it. Outputs cannot
overwrite an input. Existing outputs require explicit `--overwrite`.

The registry maps `tasks.<task>.incumbent` to a strategy, frozen source hash and
optional config hash; its frontier retains objective-dependent alternatives.
`p0-review/registry.json` initializes the production baseline and provisional
frontier, not a claim that baseline is the universally best predictor. Its original
baseline report hash is immutable. Update task incumbents only after reviewed
promotion; continue comparing against the immutable baseline to detect drift.

For a future candidate versus an older frozen incumbent, add
`--reference-report OLD_REPORT --reference-run-strategy auto
--baseline-strategy incumbent --candidate-strategy auto`. This permits the same
strategy name at different versions. Both reports must be complete, have exactly
the same case inventories and input hashes, and preserve per-strategy provenance.
The reference alias is compared with the registry's original strategy/version.
Different objective configurations are distinct candidates; never silently merge
different source/config versions by strategy name alone.

A policy object has `status: provisional|official`, optional `source` and
`source_sha256`, and `task_metrics` mapping task names to metric-name/boolean
higher-is-better maps. An official declaration requires its written source hash.
Official scoring arithmetic/normalization must be supplied by a separately tested
offline scorer; this implementation does not invent an unknown formula.

A manifest supplies `comparison_pool` and `expected_cases` (task/name/seed objects).
Confirmation additionally requires `stage: confirmation`, `predeclared: true`,
`previously_unopened: true`, `frozen_source_sha256`, and `objective_policy_sha256`
(SHA-256 of policy JSON with sorted keys and compact separators). Previously
exposed seeds 42/314/2027/4099/8081 cannot establish new confirmation. A complete
inventory can detect entirely omitted families, unlike a report's observed rows.
P0 review manifests are explicitly retrospective development inventories.

Optional `--prediction-evidence FILE` accepts an evaluator-owned object with the
original `report_sha256` and `cases`: task/name/seed/strategy/data_sha256 plus
nonempty aligned `truth` and `prediction` arrays. Classification labels remain
opaque strings. Independent confusion matrices, class supports/recalls, macro F1,
MAE/RMSE and ARI can be calculated without ML dependencies. Existing saved metrics
must agree. AMI must be supplied by an offline scorer; it is not approximated.
Truth and predictions are evaluator artifacts and must never enter selection APIs.
The helper enriches a copy and does not modify production or historical files.

Offline reproduction, using a new output directory:

```powershell
.\.venv312\Scripts\python.exe -m unittest discover -s reports/grade-evaluation -p test_*.py
.\.venv312\Scripts\python.exe reports/grade-evaluation/run_p0_review.py --output-dir reports/grade-evaluation/p0-review-repeat
```

The runner reviews four existing comparisons, checks six cross-objective input
hashes, creates a read-only incumbent/frontier registry and preserves old outputs.
No fits, network requests or production imports occur. P0 outputs contain family
uncertainty, missing metrics, severe alerts and task-specific recommendations.

Validation: 66 focused tests cover legacy explicit diagnostics, removed budgets,
task independence, family weighting, rank direction/ties, severe alerts, zero-loss
references, missing scenarios, malformed/partial evidence, actual 60-second
deadlines, recovery semantics, frozen incumbent comparisons, evidence sidecars,
official objective declarations and promotion preconditions, and output protection.
