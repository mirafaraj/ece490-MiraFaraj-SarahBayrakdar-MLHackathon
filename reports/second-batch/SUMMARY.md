# Second targeted batch: results and retention decisions

Starting clean checkpoint: `d770d8bbea25820dd0f4cbd231786e4427c7ca30`.
The strongest measured improvement is optional lazy parent imports: **1.758s
median paired saving with unchanged predictions and selection in all 20 pairs**.
This passes the evaluator's engineering criterion, but did not increase stronger
model completion. The other experiments exposed important failure modes and do
not support changing production defaults. No hackathon points can be estimated
without official scoring and assessment-environment verification.

## Before and after

| Investigation | Predictive quality | Runtime | Validity / recovery | Decision |
| --- | --- | --- | --- | --- |
| Current routed vs optional lazy imports; 60 randomized CLI calls | Identical predictions, selected candidates and validation scores in all 20 pairs; wider/25 mean BA .80739 for both vs auto .88424 | Paired median saving 1.758s; parent router imports median 1.828s to effectively zero | 60/60 valid, max 22.531s; one baseline recovery per routed variant at wider/25; no external timeouts | Promising engineering change for independent review; not yet a proven deadline-quality gain |
| Separate lazy vs lazy + refit admission; 16 CLI calls | Wider/25 accuracy .89167 to .79167, BA .88424 to .75204, F1 .87571 to .75392, minority recall .86486 to .64865; all five pairs deteriorate. Wider/50 predictions identical in three pairs | Wider/25 median 19.859s to 13.125s; wider/50 21.782s to 21.485s (no claim from this small difference) | 16/16 valid, max 21.891s; zero fallback/recovery or implementation failures | Reject fixed admission heuristic as a quality improvement; keep disabled, with failure evidence preserved |
| One vs two classification holdouts; 36 new development cases, balanced weights, BA objective | Accuracy .80556 to .81500; BA .68802 to .68561; F1 .65450 to .65806; minority recall .60662 to .58683. Worst paired BA decline .175 | Median selection work .839s to 1.671s | 132/132 outer candidate fits and six bounded selector calls valid, no fallback. Under a separate 1s selection cap, two folds exclude incomplete histogram | Do not promote two folds: benefit is inconsistent and cost approximately doubles |
| Clustering current vs no indicators + K-means; fresh random masks | Mean ARI .14500 to .45571; all 12 improve | Mean direct fit .04158s to .01825s; excludes CLI startup | All 192 clustering calls valid; zero repairs/exceptions | Promising scenario-specific result, not a universal policy |
| Same clustering alternatives; fresh group-informative masks | Mean ARI .43180 to .42504; worst loss .29398. Indicator-free geometry alone falls to .37659, worst loss .45235 | Mean direct fit .03508s to .02083s | Same 192-call study; 32/32 repeated policy prediction hashes match | Reject blanket indicator removal; retain successful covariance/representation configurations |

BA, accuracy, F1, ARI and AMI remain provisional scenarios. Full per-case results,
all three classification objectives, AMI, support vectors, exclusions, selection
frequencies, retrospective candidate gaps and raw arrays are in the linked area
reports and JSON. Direct-fit times are not end-to-end deadline certification.
Repeated calls on related generators are not independent unseen dataset families.

## What caused the losses

**Runtime:** importing modeling dependencies in the parent is a confirmed,
avoidable component of routed startup. Lightweight configuration loading removes
that cost while leaving spawned workers and atomic baseline checkpoints intact.
The original 60-call study still missed one wider/25 refit in each routed variant.
The separate, predeclared admission heuristic correctly used the actual parent
deadline but was overconservative: it rejected histogram in five cases where the
unguarded model finished and was better. A speed saving is insufficient when
every measured quality scenario deteriorates. Do not tune its factor or reserve
using these results. Host timing changed since batch one; historical timing
differences cannot be credited to the new import flag.

**Clustering:** all eight historical indicator-free losses were reproduced.
Isolating only GMM random initialization changes the missing-axis case from
ARI .679 to 1.000; K-means also reaches 1.000. This is evidence of a GMM basin
effect in that case. However the bad grouping has *better* within-representation
BIC (939.61 vs 1116.33), and all 312 fitted GMM candidates converged. Convergence,
more starts and minimum BIC do not establish correct group recovery. Severe curved
and correlated losses remain when GMM is removed, while new informative masks
demonstrate useful information in missingness. Both initialization/density-model
mismatch and representation loss matter; no single GMM explanation covers all
eight cases. No cross-representation BIC comparison or reference-label production
rule was introduced.

**Classification:** three new seeds do not replicate the earlier mean BA gain
from two holdouts. Only six of 36 BA selections change; mean retrospective BA
gap to the best evaluated family widens .00883 to .01124. The prior 39.33-point
rare-multiclass accuracy loss reproduces exactly, and a new seed shows a 40-point
loss: one and two folds both choose weighted linear. The second fold is therefore
not the cause of that failure. Tiny minority supports and an interaction the
linear model cannot represent remain plausible contributors; this batch fixes
weights and cannot newly separate weighting from family mismatch. Six additional
rare-mechanism probes show BA .58936 to .63339 but accuracy .94074 to .92741;
one changed selection explains the whole gain. This does not justify a blanket
weighting change or an outer-accuracy veto on a BA policy.

## Implementation and safeguards

- Optional CLI `--lazy-router-imports`; default off, routed only. The config loader
  is lightweight and preserves strict validation. Added parent import/config
  phases are emitted only with timing diagnostics.
- Optional classification-only `experimental.refit_admission`, default false.
  Fixed training-fit-cost estimate and actual parent deadline; incompatible
  experimental portfolios are rejected. This failed quality test remains
  experimental, not recommended for deployment.
- Optional observational GMM diagnostics record initialization, convergence,
  iterations, within-space BIC and component weights without changing fitting.
- Existing estimator families, classification holdout/weighting controls and
  clustering controls are retained. Regression models and production defaults
  are preserved. Evaluation labels enter offline reporting only.

Separate source ZIPs preserve the initial import study and later admission
version. Source/config/fixture hashes were checked before runs; classification
and clustering retain their own frozen protocol/source records. All heavy studies
ran serially. No full confirmation campaign, commit or push was performed.

## Verification

- **18/18 default-equivalence probes pass** against the starting checkpoint,
  including baseline, auto and routed tasks and the protected regression cases.
- Independent standard-library reconstruction passes 402 classification prediction
  records, 264 training-only selections, 44 input/split records, all 192 clustering
  ARIs and all 76 CLI prediction metric records. This is arithmetic/evidence
  verification, not independent rerunning of model studies or AMI derivation.
- Byte comparison preserves all 135 tracked files outside the seven intended
  edited modules, including evaluator files, historical evidence, configuration,
  dependencies and preprocessing. AST comparisons preserve the baseline model
  body and supervised estimator definitions. HEAD and the staging area remain
  unchanged. Windows CRLF is intentionally preserved under `* -text`; whitespace
  verification uses `core.whitespace=cr-at-eol` without changing repository config.
- The repository-wide run passed 216 tests and exposed one test-only exact-float
  assertion (`-.49999999999999994` versus `-.5`) in the new independent ARI check.
  Corrected that assertion to numerical tolerance; all three evidence tests then
  pass. Thus all 217 distinct tests have passed their relevant final checks.
  No production source changed after integration. Both original integration XML
  (including the failure) and corrected-test XML are preserved; the full suite
  was not repeated after this isolated assertion fix. Three expected warnings
  cover identical-row K-means and a deliberately missing validation class.

Evidence: [runtime](runtime.md), [admission](admission.md),
[clustering](clustering.md), [classification](classification.md), and
`experiments/second-batch/` (raw arrays, summaries, compatibility, preservation,
independent arithmetic checks and test XML).

## Prioritized next work

1. Independently review lazy imports first: strongest supported engineering gain,
   low behavioral risk. Check additional task families and assessment-like CPU,
   memory and installation conditions before considering default promotion.
2. Keep the failed admission rule off. A separately approved future study could
   checkpoint a completed cheap candidate while still allowing the stronger refit
   to attempt completion; first quantify extra-fit cost and interruption recovery
   quality. This concept is unimplemented. Freeze any new rule before fresh data.
3. Develop training-only clustering stability/representation diagnostics that
   preserve useful missingness and covariance gains. Indicators-on K-means is an
   independent-evaluation candidate (fresh mean ARI .398 vs .288 current), but its
   historical .52051 loss must remain a rejection case. Measure extra-fit cost.
4. Prioritize rare-class selector uncertainty and nonlinear interaction failures
   over indiscriminately doubling folds. Use supports and training-only evidence;
   preserve all objective scenarios until official rules are obtained. Keep
   regression unchanged pending its own separately scoped selector study.
5. Obtain official scoring, input/output/checker, size and memory limits, then
   separately approve unopened confirmation. All inspected results here become
   development evidence for subsequent changes.

Proposed local checkpoint, subject to user approval:
`checkpoint: validate lazy routing and second-batch reliability experiments`.
Do not commit or push automatically.
