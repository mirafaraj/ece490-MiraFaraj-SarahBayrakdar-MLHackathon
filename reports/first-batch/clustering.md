# First batch: isolated clustering controls and development evidence

Production defaults remain unchanged. The new controls are optional: numeric
missing indicators default to enabled, covariance defaults to full, and K-means
only defaults to false. They are accepted through the routed experimental
configuration or an explicit direct-call override. No regression path or shared
preprocessing was changed. Generic and baseline-centroid GMM restarts remain.

## Declared design and limits

`scripts/batch_clustering.py` declares six synthetic mechanism families:
spherical, correlated, unequal density, numeric/category mixed, curved moons,
and missing axes with an irrelevant 50%-missing noise column. Each uses 240 rows,
three development seeds (6101, 6103, 6107), and five conditions: no additional
missingness, random 10/30%, and feature-dependent 10/30%. Feature-dependent masks
use another observed feature plus noise, never reference labels. The missing-axis
family retains its intrinsic missing noise column in every condition.

Four isolated policies are tested: current geometry, K-means only, no numeric
missing indicators, and diagonal instead of full covariance. No combined policy
was tested. All policies use the same features, seeds, required k, K-means starts,
and BIC margin; only one factor changes per alternative. Reference ARI/AMI are
computed in the offline harness and never enter fitting or BIC selection.

The 360 direct calls took 22.50 seconds collectively. All 360 returned the required
number of groups and aligned predictions. These are small development tables,
not maximum-size, process-isolated CLI deadline evidence or unopened confirmation.
Reported wall times include metric bookkeeping and use fixed execution order;
they establish rough cost only, not statistically defensible paired speed claims.

## Before and after

| Policy | Mean ARI | Mean AMI | Mean wall seconds | ARI wins / losses / ties vs current |
| --- | ---: | ---: | ---: | ---: |
| Current | 0.48195 | 0.51501 | 0.0720 | — |
| K-means only | 0.51323 | 0.54539 | 0.0410 | 30 / 11 / 49 |
| No indicators | 0.74478 | 0.74531 | 0.0698 | 61 / 8 / 21 |
| Diagonal | 0.45101 | 0.48673 | 0.0620 | 7 / 12 / 71 |

| Mechanism | Current ARI | K-means only | No indicators | Diagonal |
| --- | ---: | ---: | ---: | ---: |
| Spherical | 0.62913 | 0.65882 | 0.84985 | 0.62121 |
| Correlated | 0.27531 | 0.23917 | 0.60976 | 0.17439 |
| Unequal density | 0.47495 | 0.48906 | 0.89438 | 0.45618 |
| Mixed | 0.81309 | 0.81309 | 0.93688 | 0.81309 |
| Curved | 0.14818 | 0.26194 | 0.35205 | 0.14819 |
| Missing axis | 0.55105 | 0.61731 | 0.82578 | 0.49302 |

Removing indicators is promising across every family mean but has eight losses.
Its worst loss is ARI -0.32086 on missing-axis seed6101 without additional
missingness. That family's intrinsic noise column still has missing values; the
current method recovers groups while the alternative enters a poorer GMM basin.
It must remain experimental. K-means and diagonal each lose as much as ARI
-0.52051 on correlated seed6101 without missingness, where full covariance is
useful. Diagonal loses mean ARI and AMI overall; reject it as a blanket replacement.
K-means-only is useful as an ablation and fallback, not a blanket replacement.

## Historical failure reproduction and causal conclusions

`experiments/first-batch/clustering-diagnostic.py` regenerates the exact inspected
router missing-axis mechanism with its original RNG consumption and writer's
column-name normalization, without writing other datasets. These are previously
inspected development diagnostics, not fresh confirmation.

| Original seed | Current ARI | K-means only | No indicators | Diagonal |
| --- | ---: | ---: | ---: | ---: |
| 42 | 0.38840 | 0.39514 | 0.69753 | 0.39138 |
| 8081 | 0.29617 | 0.38382 | 0.66947 | 0.29617 |

Seed8081 exactly reproduces the historical ARI loss of 0.08765 versus K-means.
Removing indicators while preserving full covariance improves both inspected
cases substantially. Changing only covariance fails to fix seed8081. Thus the
missing-indicator representation contributes causally to these losses, and the
density-based selector can make an already weak representation worse. A blanket
claim that full covariance is the root problem is contradicted by this ablation.

The likely mechanism is that standardized binary missingness axes become strong
geometry and GMM density signals. This explanation remains a hypothesis: the
experiments establish intervention effects, not a complete decomposition of BIC,
EM basin, imputation, and useful informative-missingness effects. Current GMM was
chosen in 65/90 small cases; no-indicators chose it in 46/90. BIC is still a density
objective, not guaranteed agreement with semantic reference groups.

## Correctness and reproducibility

The initial focused run passed 20 tests and hit one existing temporary-directory
setup permission error. Re-running that historical test with a fresh workspace
basetemp passed, giving 21 passed tests. A subsequently added route-control
precedence test is left for the parent's integrated test run. Tests verify default
versus explicit-default equivalence, optional settings, required k, malformed
controls, unchanged inputs, ignored supplied labels/test rows, and observational
phase timing. The parent also compares defaults against frozen checkpoint code.

Twenty repeated missing-axis configurations reproduce prediction hashes exactly.
The diagnostic initially failed because intercepting the generator bypassed its
normalization from mixed integer/string column names; the harness was corrected
to match the original writer, and all 28 diagnostic/repeat runs completed. This
was a harness setup error, not a production model failure. JSON outputs retain
source/data hashes, selections, BICs, dimensions, validity, times and external
metrics. No historical reports or evaluator files were edited.

## Next actions and retention criteria

1. Retain indicator removal as an opt-in candidate. Investigate its eight losses,
   particularly the -0.32086 missing-axis case, before proposing any policy.
2. Test truly informative missingness mechanisms and larger/noisier mixed tables;
   feature-dependent masks here are not a sufficient proxy for all such cases.
3. Only after component review, predeclare a bounded combined-policy experiment
   (representation with a training-only selection signal). Do not use external
   ARI, dataset names, or inspected reference labels in model selection.
4. Preserve useful full-covariance restarts and the completed K-means safeguard.
   Reject blanket diagonal/K-means promotion; document worst-case loss in any
   future promotion decision. Require independent family evidence, official
   contract/metric alignment and 60-second CLI validation before promotion.

No production model was promoted and no full confirmation campaign was run.
