# Latest independent model evaluation

**Keep the useful model improvements.** Current evidence supports a regression improvement, classification objective alternatives with different tradeoffs, and a narrower clustering route needing work. It does not support discarding every improvement because one ordinary-accuracy case loses. Default assessment is now task-aware; strict no-drop remains an optional diagnostic.

## Evidence actually evaluated

- Completed `experiments/router-comparison.json`: **54/54 valid full CLI runs**, 9 families × seeds 42/8081 × baseline/auto/routed, zero emergency fallbacks or checkpoint recoveries. Source stable `c7904fae62eb7f4e81b521fff47c83cd511469c5bcc15d25413d7e12ec90e609`, 50-second internal budget and 60-second external timeout.
- New controlled accuracy-objective experiment: **12/12 valid full CLI runs**, same six classification input hashes, baseline versus routed with explicit accuracy configuration; zero fallbacks. Source/config stable `e67b8bbe524e8cc450e55a770ba42224f0286f54d686ca847fdf7a3707141b53`. The evaluator independently reproduced both fingerprints from the same production files plus the explicit objective override, and verified the override SHA-256. The different fingerprint reflects configuration, not an unexplained code change.
- Completed `experiments/selector_stability.json`: **148/148 valid offline direct calls**, frozen source. Its 120-second budget and overlapping outer splits make it a development diagnostic, **not** 60-second CLI compliance.
- Independently reproduced **108 aggregate fields** across `router-vs-baseline.json`, its accuracy variant, and `router-vs-auto.json`, from raw rows (18 paired cases per summary). SHA-256 provenance, paired input hashes, validity and score directions were checked. New JSON evidence is in `latest-independent-evidence.json`.

Seeds 42, 314 and now 8081 are exposed; all are development evidence for future tuning. Historical reports still call 8081 confirmation because it was predeclared before that run, but it cannot confirm subsequent choices based on these results.

## Current default routed strategy against baseline

| Task | Measured result across six pairs | Verdict |
|---|---|---|
| Classification | Balanced accuracy +7.313 pp on average; ordinary accuracy +0.926 pp. Accuracy: 2 wins / 3 ties / 1 loss; balanced accuracy: 3 / 3 / 0. | Useful gains with an explicit objective tradeoff; retain and compare alternatives. |
| Regression | Mean per-case relative RMSE reduction **29.393%**; 4 wins / 2 ties / 0 losses. | Preserve this improvement independently; eligible for new frozen confirmation. |
| Clustering | Mean ARI change **−0.01497**; 1 win / 3 ties / 2 losses. | Keep geometry improvements as candidates; do not promote universal geometry routing. |

The classification loss is `missing_interactions`, seed 8081: ordinary accuracy 0.85556 → 0.83889 (−1.667 pp), while balanced accuracy 0.60606 → 0.83086 (+22.480 pp). That is a concrete recall/accuracy tradeoff; whether its impact is acceptable depends on official scoring and a predeclared policy, not an averaged pseudo-grade.

Regression gains are concentrated in sparse linear targets (84.856% and 76.026% RMSE reductions) and heavy-tailed targets (9.895% and 5.580%). `missing_product_target` ties both seeds. Average reduction excluding sparse linear targets is **3.869%**. This remains useful family-specific improvement, not proof of uniform improvement across private regression datasets.

Both clustering losses occur in `missing_cluster_axes`: ARI 0.39514 → 0.38840 (seed 42, −0.00673), and 0.38382 → 0.29617 (seed 8081, −0.08765). The selected model is full-covariance GMM. Correlated axes improve 0.99545 → 1.00000 on seed 8081 and tie on seed 42; unequal density ties both seeds. The old GMM initialization defect has been addressed; **this is new evidence that density selection with missing-data geometry can still hurt reference partition quality**. It should not be described as the unfixed historical noisy-blobs defect.

## Tested alternative: ordinary-accuracy objective

The explicit accuracy objective improves both ordinary and balanced accuracy against baseline on **4 cases, ties 2, and loses none** in this development set. Mean changes are **+2.130 pp accuracy** and **+4.601 pp balanced accuracy**. The measured strict no-drop diagnostic passes. This is a concrete useful alternative, not only a suggestion to tune later.

However it does not dominate the default balanced-accuracy objective:

| Same input | Accuracy-objective minus default-routed accuracy | Accuracy-objective minus default-routed balanced accuracy |
|---|---:|---:|
| Iris, seeds 42 and 8081 | 0 | 0 |
| Sparse linear classes, seed 42 | 0 | 0 |
| Sparse linear classes, seed 8081 | +4.444 pp | +4.444 pp |
| Missing interactions, seed 42 | −0.556 pp | −2.787 pp |
| Missing interactions, seed 8081 | +3.333 pp | **−17.934 pp** |

Across all six pairs the accuracy objective gains 1.204 pp ordinary accuracy but loses 2.713 pp balanced accuracy versus default routed. Keep both policies until official alignment and new confirmation; **do not blindly switch the default**. Selecting whichever objective wins each visible external case would leak evaluation information into a non-deployable routing rule.

## Does routing buy quality beyond auto?

On the latest 18 CLI pairs, **routed and auto tie every reported classification, regression and clustering quality metric**. Routed averages 7.127 s versus auto 4.973 s (extra 2.155 s, about 43.3%), and baseline 4.715 s; routed's maximum is 9.546 s. Task means:

| Task | Baseline seconds | Auto seconds | Routed seconds |
|---|---:|---:|---:|
| Classification | 4.731 | 5.426 | 7.224 |
| Regression | 4.922 | 4.941 | 7.594 |
| Clustering | 4.492 | 4.551 | 6.565 |

Routed is **dominated on measured quality and time** by auto in this ordinary-size suite. It also implements completed-baseline recovery and explicit routing/resource rules, whose value is not measured by zero-recovery normal runs. Profile/process startup/checkpoint work rather than deleting safeguards from a single speed comparison. These are observed serial-run wall times, not randomized causal estimates. The later accuracy-objective session has different baseline times too, so its timing difference cannot be attributed causally to the objective setting.

The latest CLI suite reaches only 660 training rows and 110 input features. Fast valid runs here establish neither maximum-size compliance nor official hardware/memory performance. Profile checks cover provisional raw-data semantics, not the unprovided official schema.

## Selector study: concrete lessons

Independently paired the offline scores, respecting their higher-is-better convention:

- Routed versus baseline: **10 wins / 5 ties / 1 loss** over 16 task/split pairs. The loss is high-cardinality classification, split 314, balanced accuracy −3.990 pp. This is outside the newer nine-family CLI suite and remains a useful generalization warning.
- Unrestricted search versus routed: **16 ties**. These cases provide no demonstrated quality penalty from current eligibility filtering, but do not prove safety on larger/excluded shapes.
- Nonzero near-tie tolerance versus current zero default: **15 ties / 1 loss**; sparse linear classification loses 3.889 pp on split 42. Do not promote the tested tolerance as a stability fix from this evidence.
- Opt-in categorical routing versus current routing: **2 ties**. No measured reason to enable the categorical extension globally.

Retrospective best-candidate gaps and validation-versus-outer optimism are diagnostics only; using outer labels to select the best candidate would be leakage. No direct-call result replaces a full CLI deadline test.

## Next actions ranked by useful evidence

1. **Preserve the regression improvement and both classification objective policies.** Obtain the official metric/normalization, then freeze one objective or a training-only rule. A secondary-metric loss budget must be declared before new confirmation; default budgets remain zero, so existing tradeoffs stay visible and unresolved rather than silently called harmless. Do not discard useful regression gains while classification is reviewed.
2. **Fix missing-axis clustering selection first.** Compare current GMM route with K-means on the missing-axis family at 0/10/30% missingness, correlated and unequal-density controls, plus mixed/curved controls from earlier development. Ablate missing indicators and covariance selection independently. Proposed engineering gate: no missing-axis family mean loss, preserve correlated gains, zero new invalid/fallback cases; freeze the rule before unopened confirmation. These are proposed gates, not contest scoring thresholds.
3. **Measure routing's recovery benefit and overhead.** Use serial paired full-CLI timing after freeze, with short budgets and deliberately expensive permitted inputs; record useful completed-baseline recovery quality, emergency fallback rate and external wall time. Seek lower overhead with unchanged selected model/metrics. Keep recovery safeguards unless the comparison establishes their cost and value.
4. **Target the residual modeling gaps.** Missing-product regression ties both CLI seeds, high-cardinality selection loses on one offline split, and the tested near-tie policy worsens one case. Compare repeat training-only holdouts and narrowly scoped representations, changing one factor at a time. Acceptance must include worst-family loss and error/fallback checks, not only aggregate gain.
5. **Finish assessment compatibility and unopened validation.** Official input/output/profile/schema/metric mapping and all practice datasets; permitted maximum-size/memory tests on the assessment environment; clean installation/CI; frozen fresh seed/family confirmation; team/repository/submission commit. `STAGE_REVIEW.md` contains the stage-by-stage handoff. No official grade estimate is defensible yet.

## Evaluator behavior and validation

`analyze.py` now defaults to **tradeoff-aware per-task assessment**, with `--baseline-strategy`, `--candidate-strategy`, `--classification-objective`, `--accuracy-loss-budget` and `--balanced-accuracy-loss-budget`. Outcomes distinguish equivalent, Pareto improvement, scalar improvement, tradeoff, harmful and inconclusive evidence. Reliability/provenance failures still block; a quality loss in one task does not discard improvements in other tasks. Strict `--enforce-accuracy-gate` remains available for the earlier no-drop diagnostic and returns nonzero on reject/insufficient evidence.

```powershell
.\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py experiments/router-comparison.json --candidate-strategy routed --output reports/grade-evaluation/latest-routed-vs-baseline.json
.\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py experiments/router-comparison.json --baseline-strategy auto --candidate-strategy routed --output reports/grade-evaluation/latest-routed-vs-auto.json
.\.venv312\Scripts\python.exe reports/grade-evaluation/analyze.py reports/grade-evaluation/accuracy-objective-development.json --candidate-strategy routed --classification-objective accuracy --output reports/grade-evaluation/latest-accuracy-objective-vs-baseline.json
.\.venv312\Scripts\python.exe reports/grade-evaluation/summarize_latest.py
.\.venv312\Scripts\python.exe -m unittest discover -s reports/grade-evaluation -p test_promotion.py
```

**33 focused stdlib tests pass**, including tradeoff preservation, predeclared budgets, independent task decisions, equivalent-versus-improved labels, quality/time dominance, malformed/partial artifacts, supplied-summary mismatch blocking and strict compatibility. The evaluator ran only artifact analysis and its own stdlib tests; the parent chat ran the new controlled 12-CLI experiment. No production code or shared test files were changed by this evaluator.

This report supersedes earlier blanket accuracy-based rejection advice in `GUARDED_REVIEW.md` and the historical assessment. Those files/results remain preserved as prior evidence; the user's latest tradeoff-aware preference controls current decisions.
