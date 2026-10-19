# Development-agent roadmap after P0 evaluator correction

## Current decision

Retain useful candidates and do not change production defaults from the existing
small reused benchmarks. Ordinary accuracy and balanced accuracy are neither
protected nor official by assumption. Official formulas/profile semantics remain
unavailable, so no total grade or guaranteed score increase can be calculated.

Evidence reviewed: the frozen 54-run router comparison, the 12-run accuracy-objective
experiment and the historical 48-run broad comparison. No new model experiment
was performed. The 148-run selector study remains a separate 120-second direct-call
diagnostic, not 60-second CLI evidence. Current-source verification previously
recorded 153 integrated tests plus eight strengthened invariance checks.

The new review lives in `p0-review/`. It preserves historical files. All task
promotion decisions currently lack adequate new-family confirmation and alternative
metrics: current comparisons have three families per task, historical ones four;
macro F1, MAE and AMI are absent. This is insufficient evidence, not proof that the
useful candidates should be discarded.

## Prioritized work (not implemented here)

### P0 continuation — collect adequate independent evidence

**Problem/evidence:** current rows are repeatedly inspected; old and current source
versions differ. Existing confirmation labels no longer establish freshness.

**Change:** capture evaluator-only predictions/references and missing metrics in
the offline harness, using the new hash-bound sidecar interface or equivalent
independently checked metrics. Keep original baseline and incumbent snapshots.
Use `--reference-report` for a challenger versus a prior frozen version, and
register objective/config alternatives separately. Predeclare case/pool manifests.

**Validation:** reconcile saved confusion matrices, class supports, macro F1,
MAE/RMSE and clustering references on hand-computed fixtures; verify changing
references never changes selection/predictions. Add at least eight distinct
mechanism families per task and three fresh confirmation seeds after freezing.
Include unseen real tables when available; related synthetic mechanisms and
multiple real-table splits are not automatically independent datasets.

**Acceptance:** full inventory/hash agreement, no hidden omitted failures, positive
family-level evidence for official score or provisional robust ranking, and explicit
severe-loss review. Missing evidence means retain/insufficient, not promotion.

**Why/risk:** reduces benchmark overfitting and false improvement claims. Offline
validation is more expensive but must not increase assessment-time model work.

### P1 — isolate classification weighting and selection effects

**Evidence:** on `missing_interactions` seed8081, balanced-weight logistic regression
scored accuracy0.83889/BA0.83086. The accuracy override chose unweighted Extra Trees,
scoring0.87222/BA0.65152: +3.33 ordinary-accuracy points but -17.93 BA points.
Aggregate metrics imply minority recall81.8% to30.3%, not a small neutral change.
The earlier accuracy instruction motivated this experiment; it changed both
weights and selected model and did not alter production defaults.

**Change/experiment:** decouple class weighting from selection objective. Evaluate
three existing estimator families times balanced/unweighted fitting on identical
inner folds; score the same predictions under accuracy, BA and macro F1. Include
binary/multiclass, 50/80/95% majority prevalence, 0/20% missingness, linear and
interaction signals, and relevant/irrelevant categories. Use two inner stratified
holdouts at the existing total selection budget. Log all candidate scores, class
recall and final selected-estimator frequency.

**Why:** isolates causal contributors and tests generalization beyond one objective.
**Accept:** positive fresh-family official-score evidence or supported robust
scenario ranking. Preserve conflicting objective policies as alternatives; do not
route by visible dataset names or external labels. **Reject:** leakage, unreproduced
selection claims or no useful evidence beyond known cases. **Risks:** extra fitting,
rare-class variance and selection overfitting. Do not add estimators/threshold grids.

### P1 — address missing-axis clustering losses

**Evidence:** current full-covariance GMM loses on both missing-axis cases, worst
ARI-0.08765; overall ARI change-0.01497. The prior initialization bug is fixed.

**Change/experiment:** compare current routing with K-means-only, then separately
ablate numeric missing indicators and full-versus-diagonal covariance. Test
spherical, correlated, unequal-density, mixed, curved and missing-axis families;
0/10/30% random and feature-dependent missingness; three development seeds per
condition. Only test a combined policy after its component effects are understood.

**Why:** addresses a reproduced representation/selection weakness while retaining
useful GMM restarts. **Accept:** improved fresh external objective/robust ranking,
documented worst-family losses and valid required-k outputs. **Reject:** a selector
using reference ARI or a rule keyed to the benchmark. **Risks:** removing indicators
can lose informative missingness; diagonal covariance can lose correlated groups.
Keep gains independently; do not adopt a blanket clustering replacement.

### P2 — preserve regression diversity and improve selection stability

**Evidence:** current mean relative RMSE reduction29.39%, four wins/two ties; only
3.87% excluding sparse-linear targets. Missing-product cases tie. A separate
high-cardinality classification split loses BA3.99 points in the selector study.

**Change/experiment:** retain Ridge/tree/boosting candidates. Compare current single
holdout with two fitting-only holdouts under equal total selection time. Use
additive, interaction/product, heavy-tail, heteroscedastic, mixed-category and
missing-data mechanisms. After that comparison, test histogram leaf counts15/31
as a separate factor. Measure RMSE, MAE, dummy-relative loss, tail and missing-row
errors, selection frequency and final refit time.

**Why:** targets remaining failures without discarding inexpensive linear gains.
**Accept:** supported fresh-family official/provisional improvement, including
results excluding linear cases. **Reject:** benefits confined to one inspected
family or increased invalid/deadline failures. **Risks:** repeated validation and
larger trees consume budget and can overfit. Keep categorical counts disabled and
near-tie tolerance zero unless a new experiment reverses existing negative evidence.

### P2 — establish the value of checkpoint-first execution

**Evidence:** routed matches auto in18 quality comparisons but averages7.13s versus
4.97s. No recovery was exercised by those normal runs; recovery tests establish
capability rather than expected score gain. Latest inputs max660 rows/110 features.

**Change/experiment:** instrument load/profile/baseline/selection/refit/serialization.
Randomize serial paired strategy order, repeat timings three times, and test
10/25/50-second internal budgets on normal and expensive permitted shapes. Record
process-tree peak memory, external wall time, completed-baseline recovery quality
and emergency fallback frequency. Optimize the largest measured overhead first.

**Accept:** an overhead-only change preserves deterministic predictions/selection
and reduces paired runtime; recovery tradeoffs must demonstrate useful deadline
behavior. Every official-size case must finish<60s; target55s headroom. Memory
acceptance awaits published limits. **Risks:** removing safeguards may improve
ordinary timing but lower expected quality during deadline failures.

## Final promotion and submission

When written rules arrive, implement and hand-test exact official scoring and
profile normalization before choosing task champions. Freeze candidate/source,
policy, comparison pool and manifests before unopened confirmation. Positive
95% family-bootstrap evidence may recommend promotion only with verified incumbent
and immutable-baseline comparison, severe-regression review, official checker,
clean installation, maximum-size/memory evidence and fallback-quality checks.

Resolve contract/practice data, run remote CI, fill team/repository details and
freeze the submitted commit. Do not rebuild completed router/leakage/model-card
work. Update task incumbents only after a reviewed promotion; preserve source and
report hashes. No ML changes in this roadmap were implemented by this task.
