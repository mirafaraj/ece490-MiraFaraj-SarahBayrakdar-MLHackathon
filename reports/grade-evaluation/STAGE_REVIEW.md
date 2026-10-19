# Progress review and remaining execution order

Reviewed 19 October 2026 against the completed chat "Build dataset analysis
program", PHASES.md, reports/router_validation.md and MODEL_CARD.md. This is an
independent review; production files and the implementation chat were not changed.

## Overall assessment

The project is a tested provisional AutoML program with a substantial evaluation
harness. It is ready for focused policy experiments and environment verification.
Official submission compatibility and maximum-size performance remain unverified.
More architecture is not the immediate priority: measure and fix observed policy
failures, then close operational gaps. Retain useful improvements per task.

## Status by stage

| Stage | Completed evidence | Remaining work / recommendation |
|---|---|---|
| 1: contract and objective | Modular entry point, provisional CSV/JSON interface, task dispatch and profiling | Official input/output/profile schemas, scoring normalization, checker, nine practice datasets and limits remain unavailable. Map written rules to actual fields and resolve objectives when supplied. Do not infer ordinary versus balanced accuracy from imbalance alone. |
| 2A: routing and resources | Versioned training-only rules, explicit objectives, candidate eligibility, strict configuration and trace metadata | Optional routing is implemented. CLI evidence does not show a quality gain versus auto, and optional exclusions were not activated by the small study. Test eligibility boundaries and large cases rather than adding rules without evidence. |
| 2B: completed-result recovery | Real timeout/crash/refit recovery and corrupt-checkpoint tests; completed baseline reuse | Keep this correctness improvement. Measure its time cost and how often it prevents emergency predictions on larger permitted cases. Small CLI suite had no failures, so it does not quantify a normal-operation quality benefit from checkpointing. |
| 3A: correctness and leakage | 153 integrated tests; eight strengthened post-prediction invariance checks; independent reviews | Record these as completed, not pending. Rerun relevant checks after changes to fitting/selection. Eight checks were a focused rerun, not 161 distinct integrated tests. |
| 3B: environment | Pinned versions and working Windows Python 3.12 environment | Clean isolated install, remote CI, assessment hardware, maximum rows/columns/cardinality, memory and external timing remain unverified. Bundled shared packages are not a clean-install result. |
| 4A: selector stability | 148 valid offline direct-call runs, eight families/two overlapping splits; routed 10 wins/5 ties/1 loss vs baseline under provisional objectives | High-cardinality split 314 loses about 0.03990 balanced accuracy. Diagnose validation-to-outer ranking and candidate uncertainty before enlarging the portfolio. Diagnostic budget was 120 seconds; this is not 60-second CLI compliance. |
| 4B: categorical candidate | Implemented, fold-tested, disabled by default | Its enabled selector did not select it on the two measured splits and had one tie/one loss against baseline. Preserve the experiment; broaden evidence only if category-driven failures justify the effort. Fast fitting alone is not a reason to enable it. |
| 4C: near-tie policy | Configurable; zero default; positive tolerance explored | Positive tolerance changed one classification choice, losing 0.03889 balanced accuracy for roughly 0.066 seconds saved. Keep zero unless a new frozen development experiment demonstrates a worthwhile tradeoff. Do not relabel this completed negative experiment as missing implementation. |
| 4D: frozen comparison | 54 valid CLI runs, zero fallbacks, stable source/config; nine families, seeds42/8081, three strategies | Routed and auto tie quality in all18 pairs; routed mean7.127s versus auto4.973s. Classification has objective tradeoffs; missing-axis clustering has material ARI loss. Promote task policies separately. All inspected seeds, including8081, become development for future changes. |
| 5A: model card | MODEL_CARD.md and current verification report exist | Update only after measured changes; distinguish historical evidence, current configuration and unsupported official claims. |
| 5B: submission | Local runnable project and dependency lock | Official checker/practice results, clean install/CI, team names, public repository access and final frozen deadline commit remain outstanding. Publishing is a separate step. |

## Prioritized next steps

1. **Adopt the tradeoff-aware review.** Keep both classification metrics and each
   task's gains. A smaller ordinary-accuracy loss with a large balanced-accuracy
   gain is a tradeoff to assess against the objective, not an automatic deletion.
   No official score is calculable without its normalization. Set material-loss
   budgets before fresh confirmation; the present defaults do not silently permit
   loss.
2. **Run one objective-alignment ablation.** Compare the frozen current routed
   classifier with its accuracy-objective variant on identical existing input
   hashes. This review is running that 12-CLI development experiment separately,
   after the implementation chat completed. Change only the objective/associated
   weight policy, not preprocessing, thresholds or default configuration. Compare
   both metrics and every case, including equal-class and multiclass controls.
3. **Target the reproduced missing-axis clustering failure.** Confirmation ARI
   fell from0.3838 to0.2962. Test numeric scaling, missingness-indicator contribution
   and GMM eligibility/selection one factor at a time, with complete-data, noisy,
   correlated and mixed-feature controls. Keep successful restart improvements
   while avoiding a blanket policy replacement. Any production routing rule must
   use unlabeled features, not the case name or reference ARI.
4. **Preserve measured regression gains and improve selection reliability.**
   Evaluate a conservative validation margin or a second small training-only
   holdout under the same total selection budget. Test high-cardinality failure
   cases and mixed/missing regression with controls. Selection must not use the
   external best-candidate diagnostic as a deployable oracle.
5. **Close independent installation and resource evidence.** Use an isolated
   environment/checkout; never modify the working .venv312. Follow with official
   maximum shapes and memory limits once published. Measure complete process-tree
   wall time/memory and fallback quality, not just fit time or valid output.
6. **Freeze and confirm, then submit.** Freeze source, policy, objective, any
   tolerances and evaluation manifest before unopened new cases. Inspect once,
   decide per task, and use another fresh set if tuning resumes. Run the official
   checker and all nine practice tables, fill team/repository details, and record
   the final submitted commit.

## How to interpret experimental timing

The historical default comparison was sequential as reported.
The new objective experiment is a separate session; compare its paired strategies
locally and treat cross-session timing changes as observations, not causal proof.
Source hashes differ when an explicit configuration is fingerprinted, so match
production source and input hashes and record the intended configuration change.
No new test result can establish a private-dataset guarantee or official points.

## Additional controlled experiment completed in this review

The accuracy-objective experiment completed all 12 actual CLI runs successfully,
with stable source/config hashes and no fallbacks: three classification families,
two already-exposed seeds and baseline/routed strategies. Against baseline,
accuracy-oriented routed improves both ordinary and balanced accuracy on four
cases and ties on two; there are no measured losses on these six comparisons.

Against the previously measured balanced-objective routed policy on identical
input hashes, four cases tie or improve on both metrics. Missing-interaction
seed 42 loses 0.56 percentage points ordinary accuracy and 2.79 points balanced
accuracy. Seed 8081 gains 3.33 points ordinary accuracy but loses 17.93 points
balanced accuracy. Thus an accuracy switch is a useful candidate, not a universal
improvement over the existing balanced router. Keep both objective policies for
explicit tradeoff analysis; no production default changed.

The current production fingerprint was independently recomputed and still equals
the prior freeze, c7904fae62eb7f4e81b521fff47c83cd511469c5bcc15d25413d7e12ec90e609.
The new fingerprint includes the deliberate explicit configuration change. These
are development observations on previously inspected rows, not fresh confirmation.
