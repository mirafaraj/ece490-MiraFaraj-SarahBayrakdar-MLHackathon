# Classification second batch — predeclared development protocol

Production source, defaults, families and regression remain unchanged. Existing
optional one/two-holdout controls are sufficient for this investigation. Official
metrics remain unknown. No accuracy/BA veto or global weighting expansion is added.

Replicate the first twelve-case design on development seeds 419, 733 and 1201.
Use balanced weights and the existing Extra Trees, logistic and histogram models.
Fit each family once on each of the two folds and once on all training rows; reuse
these predictions for accuracy, balanced accuracy and macro F1 selection scenarios.
Compare one/two holdouts using the same 20-second total selection cap. The primary
study is exhaustive offline work; measured cumulative fold-time replay records
cap exclusions. Separate actual bounded-selector calls at 20/50-second requested
budgets test existing shared-cap behavior; an additional temporary 1-second cap
tests order and exclusion behavior. These are direct calls, not CLI proof.

Six separate rare multiclass probes cross three seeds with multiplicative/XOR-tail
and localized-tail signals. Reproduce the already inspected seed317/case10 loss,
and repeat seed419/case10 exactly. These are development probes, not independent
confirmation families. Keep minority-class supports explicit.

Save complete input tables, training labels, split indices/supports, reference and
prediction arrays, input/split/prediction SHA256, inner/outer scores, exclusions,
selection frequencies and retrospective gaps. Independently recompute all raw
metrics using confusion-matrix arithmetic. Outer labels enter reporting only.
No outer-score choice is eligible for production routing.

No promotion. Advance two holdouts only if benefit replicates across new seeds
and losses are disclosed; reject incomplete-fold competition, leakage, unexplained
non-reproducibility or assumptions that one provisional metric is official.

Reproduce after obtaining the serial CPU slot:

```text
.venv312/Scripts/python.exe scripts/second_classification.py --output experiments/second-batch/classification.json
.venv312/Scripts/python.exe -m pytest tests/test_second_classification.py tests/test_batch_classification.py -q
```

Output is exclusive: choose a new path for subsequent runs. Completed identities
are cached under ignored experiments/work with a source/protocol-derived name;
an interrupted run can resume without repeating completed fits. Final evidence
uses compact JSON and includes this pre-fit protocol snapshot.

## Measured findings

The fixed study completed in **130.96 seconds offline**. All **132 outer candidate
fits** returned valid predictions without fallback (36 replication cases, six rare
mechanism probes, one inspected reproduction and one exact repeat, each with three
families). The six actual bounded calls also returned valid predictions with no
fallback. All 402 saved prediction/reference records passed independent confusion-
matrix metric recomputation. Seed419/case10 reproduced every input/split hash,
candidate prediction hash, validation score and selection exactly. This checks
determinism for one case only; it is not cross-machine confirmation.

The three new seeds do **not** replicate a robust balanced-accuracy improvement
from doubling holdouts. Below are mean outer metrics on the 36 new development
cases; weights are fixed balanced throughout, and higher is better.

| Inner objective | Holdouts | Accuracy | BA | Macro F1 | Minority recall |
|---|---:|---:|---:|---:|---:|
| Accuracy | 1 | .85981 | .63639 | .63630 | .43990 |
| Accuracy | 2 | .86389 | .64201 | .64434 | .44529 |
| BA | 1 | .80556 | .68802 | .65450 | .60662 |
| BA | 2 | .81500 | .68561 | .65806 | .58683 |
| Macro F1 | 1 | .83833 | .68866 | .68108 | .56999 |
| Macro F1 | 2 | .85667 | .68534 | .67760 | .53567 |

For the incumbent BA scenario, two holdouts improve accuracy by 0.94 percentage
points and F1 by 0.36, but reduce BA by 0.24 and minority recall by 1.98. BA changes
by +0.67, -1.46 and +0.07 points on seeds419/733/1201 respectively. Only six of 36
selections change: three improve all three metrics, one worsens all three, and
two have conflicting accuracy/BA/F1 effects.
The worst paired BA loss is **17.50 points**, on seed733/case8: the additional
holdout chooses histogram instead of linear, reducing BA .76389 to .58889 while
accuracy rises .80667 to .85333. The worst paired accuracy loss is 8.67 points.
The largest accuracy win, 34 points, occurs on one rare binary interaction case;
it contributes materially to the small aggregate gain.

One-holdout BA selection chooses 23 linear / 11 histogram / 2 tree models; two
holdouts choose 22 linear / 14 histogram / 0 tree models. Mean retrospective BA
gap to the best evaluated candidate **widens .00883 to .01124**. All development
candidates complete under the descriptive 20-second timing replay, so these
selection differences reflect split estimates rather than excluded candidates.

Accuracy-based two-holdout selection improves all three means on this design, but
continues to sacrifice BA relative to the BA policy. Macro-F1 selection improves
accuracy while slightly reducing BA/F1. These are competing provisional scenarios,
not evidence that accuracy is the official objective or a universal override.

## Rare multiclass losses: reproduced cause and remaining uncertainty

The inspected seed317/case10 loss is reproduced exactly: balanced trees achieve
accuracy .94667 / BA .33333 / F1 .32420, while balanced linear achieves accuracy
.55333 / BA .43779 / F1 .28923. Both holdout policies select linear for BA. Thus
the second holdout does **not** introduce or correct the 39.33-point accuracy loss.
The training-derived BA objective rewards minority recall despite a large majority
recall cost; this is a confirmed objective tradeoff, not selection-label leakage.
Both policies correctly maximize their recorded inner BA scores.

The failure generalizes to some new seeds: seed733/case10 loses **40 accuracy
points** against the tree baseline (.94667 to .54667) under both policies; BA rises
only .33333 to .35446 and F1 falls .32420 to .27413. Seed419/case10 loses 38.67
accuracy points but gains 18.78 BA points. Each has training supports428/11/11,
inner validation supports107/3/3 and outer supports142/4/4. One minority prediction
changes per-class outer recall by25 points. Weighted linear fitting, interaction
misspecification and this discrete uncertainty plausibly contribute; this batch
holds weights fixed and does not causally separate their effects. Historical
weight/family interaction evidence remains relevant; no blanket weighting change
is justified.

The six additional rare-mechanism probes give two-holdout BA .58936 to .63339 and
F1 .55625 to .55963, but accuracy falls .94074 to .92741. **One** localized-tail
case accounts for the entire change: histogram to linear raises BA by26.42 points
while losing8 accuracy points. The other five selections are identical. These
two mechanisms do not establish broad rare-class robustness. Some minority-class
recalls remain zero; full supports, recall vectors and all candidate predictions
are retained in the JSON rather than hidden by aggregate means.

## Selection cost, exclusions and validity

Median exhaustive inner work rises **.839 to1.671 seconds** from one to two
holdouts; the corresponding maxima are1.116 and2.158 seconds. The rare-probe
medians are .791 and1.597 seconds. These timings are model-selection work only,
not isolated-process startup, completed-baseline cost or full CLI runtime.

Actual direct selectors on seed419/case10 all retain linear and identical raw
predictions. At20 seconds requested budget (shared selection cap8s), one/two
holdouts take1.023/2.148s total; at50 seconds (cap20s), .897/1.693s. A separately
predeclared1s-cap stress probe takes .903/1.045s: one holdout completes all three
families, while two exclude histogram as incomplete. No incomplete scores enter
selection, all returned outputs remain valid, and no fallback occurs. Timed fits
can finish after a pre-fit cap check; worker deadlines remain the external guard.
This probe demonstrates order/headroom risk, not a recommended1s deployment cap.

## Assessment and next steps

**Keep two holdouts optional; do not promote.** Their benefit depends on objective,
seed and mechanism, while cost roughly doubles and severe errors remain. Preserve
all existing models and production defaults. This batch adds scripts, tests and
development evidence only; supervised.py, its regression path and configuration
defaults are unchanged by this classification investigation.

Next, obtain the official objective before choosing among the retained scenarios.
Develop better train-only reliability diagnostics for tiny minority supports and
validate them on fresh mechanisms before considering any selector change. Do not
silently veto BA-selected models using outer accuracy. If a multi-metric or
uncertainty-aware policy is proposed, predeclare its training-only rule and cost,
compare against these frozen controls, and expose all losses. Runtime improvements
that recover refit headroom deserve priority over adding more folds indiscriminately.

Focused checks: **16 passed** (the four new harness checks plus the12 existing
classification-control tests). The single warning deliberately tests a class
present in predictions but absent from validation truth. Test coverage includes
independent metric arithmetic, lexical labels, incomplete-fold exclusion, shared
caps, outer-label-independent decisions, default dispatch, test-feature invariance
and checkpoint recovery. No experiments or policies were promoted, and no commits
or pushes were made.

Evidence: `experiments/second-batch/classification.json`. The artifact retains the
exact pre-fit protocol snapshot and its hash, source hashes, full raw arrays and
per-case support/timing/selection details. All inspected rows are development
evidence for future work, not unopened confirmation.
