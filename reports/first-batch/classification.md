# Classification first batch: development protocol and results

Official classification grading remains unknown. Production defaults remain
unchanged. These are development findings, not promotion or confirmation evidence.

## Design declared before running

Twelve cases use seed 317 and 600 rows each: 450 training and 150 outer evaluation
rows. Binary and three-class problems are crossed with 50%, 80% and 95% majority
prevalence, with two cases per combination. Other factors rotate through 0%/20%
missing numeric cells, linear/interaction signals, and relevant/irrelevant
categories. This bounded design is not a full factorial, and the cases are not
independent mechanism families. Thresholds on a noisy latent score set class
proportions. Two inner stratified 25% holdouts use seeds 317 and 318. The first
two cases repeat exactly to check reproducibility. Overlapping splits are not
fresh datasets or unopened confirmation.

The existing Extra Trees, logistic regression and histogram boosting candidates
are fitted with balanced and unweighted class weights on identical folds. All
statistics are fitted separately inside each fitting fold. The same predictions
are scored for ordinary accuracy, balanced accuracy, macro F1 and per-class recall.
Outer evaluation labels enter only offline reporting after candidate decisions.
No estimators or threshold grids are added.

The offline exhaustive decomposition crosses three objectives with balanced,
unweighted and combined portfolios, and with one or two holdouts. It reports
selection frequencies, gaps to the retrospectively best evaluated candidate,
losses versus balanced Extra Trees, validity, fallback rate and elapsed time.
This exhaustive study is not bounded-selector or 60-second CLI evidence. Inner
elapsed time is reported against the 20-second production selection cap. Separate
bounded-selector tests check that all holdouts share one cap and exclude incomplete
two-fold candidates, rather than mixing one-fold and two-fold averages.

Acceptance for this batch means identifying promising changes and severe losses;
there is no promotion. Prefer effects supported across metrics and scenarios.
Conflicting objectives remain alternatives. Reject leaked labels, incomplete-fold
comparisons, unreproducible claims, default changes, and official-score claims.

The previously inspected missing_interactions table, generated with seed 8081, is
also examined. This is known development evidence. Its six candidates and two
inner folds provide the weighting/family decomposition and one-versus-two-holdout
comparison. Generated tables use an isolated temporary directory; historical
sources remain untouched. An initial protocol error used model seed 42 for this
case; the historical evaluator used 8081. Both versions are retained and clearly
identified below.

Reproduction commands, after obtaining a serial CPU slot:

```text
.venv312/Scripts/python.exe scripts/batch_classification.py --output experiments/first-batch/classification-new.json
.venv312/Scripts/python.exe -m pytest tests/test_batch_classification.py tests/test_supervised_candidates.py tests/test_routed_supervised.py -q
```

Choose a new output path: the harness refuses to overwrite an existing result or
recovery artifact.

## Measured results on the twelve development cases

The completed run took 91.58 seconds offline. All 78 distinct outer candidate fits
(72 from the new design and 6 from the known case) returned valid outputs without
fallback. Both repeated cases reproduced every candidate prediction hash,
inner/outer metric and selection exactly. This establishes limited deterministic
reproducibility, not cross-machine or new-seed confirmation. Exhaustive inner work
peaked at 4.73 seconds per case, against the configured 20-second selection cap.
These small tables do not establish assessment-time or maximum-shape performance.

Mean outer metrics are provisional; larger is better:

| Fitting weights / objective | Holdouts | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Extra Trees baseline, balanced | none | .83389 | .53905 | .54052 |
| Balanced / balanced accuracy | 1 | .81056 | .68129 | .65357 |
| Balanced / balanced accuracy | 2 | .81167 | .69095 | .66039 |
| Unweighted / accuracy | 1 | .85222 | .62764 | .63584 |
| Unweighted / accuracy | 2 | .85944 | .64167 | .64931 |
| Balanced / macro F1 | 2 | .84111 | .67108 | .65519 |
| Both weights / macro F1 | 2 | .84444 | .66329 | .65942 |

Two holdouts are promising, but not uniformly beneficial. For the incumbent
balanced-weight/BA policy, mean balanced accuracy rises 0.97 percentage points;
the mean BA gap to the retrospectively best balanced candidate falls from 1.08
to 0.11 points. Selection changes from 5 linear / 2 trees / 5 histogram cases to
8 linear / 1 tree / 3 histogram cases. For unweighted accuracy selection, the
second holdout improves all three mean metrics. For unweighted BA selection,
however, mean BA drops 0.47 points and macro F1 drops 0.71 points. The JSON retains
all policies, selection frequencies, fold metrics, candidate gaps, baseline losses,
class recalls and timings.

Fitting effects differ by family. Averaging unweighted minus balanced on the same
12 tables and same final candidates:

| Family | Accuracy change, points | BA change, points | Macro F1 change, points |
|---|---:|---:|---:|
| Logistic regression | +14.83 | -3.58 | -0.77 |
| Extra Trees | -0.11 | +0.16 | +0.33 |
| Histogram boosting | +0.61 | -0.30 | +0.23 |

Changing objective and weights together cannot isolate these effects. Ordinary
accuracy alone does not justify a blanket weighting policy.

There are severe unresolved failures. In case 10 (three classes, 95% majority,
interaction signal, relevant category), balanced BA selection chooses linear and
gets 55.33% accuracy versus 94.67% for balanced Extra Trees: a 39.33-point loss.
BA rises from .3333 to .4378, but macro F1 falls 3.50 points. The majority and two
minority recalls are 56.34%, 50% and 25%; each minority class has only four outer
rows. In case 5 (95% majority, interaction signal, 20% missingness), two-fold BA
selection still has zero minority recall. More folds do not create missing signal,
and balanced accuracy alone can reward a poor accuracy tradeoff. No policy
dominates every scenario or metric. Small class supports limit these conclusions.

## Correctness, execution and source versions

The first attempt finished 12 cases and 2 repeats, then the Windows sandbox refused
child writes to its default temporary directory during known-case generation.
No consolidated result was produced. The harness now uses an isolated workspace
temporary directory and appends each completed case to a recovery JSONL. The run
was repeated once; no historical file was replaced. This was an experiment-storage
failure, not a model-quality failure.

Before the successful retry, forced-refit review found that recovery could launch
another fit when experimental and checkpoint weights differed. Recovery now
returns the completed checkpoint after a failed optional refit, reports its actual
weight and variant, and avoids fitting again. Compatible checkpoints are also
reused on normal tree selection. Focused tests passed 28 checks before the retry.
An additional Extra Trees failure edge case is checked separately after the study.
Default calls still execute the incumbent path. Regression fitting and selection
remain unchanged. No experimental production configuration is enabled.

The initial known-case analysis used model seed 42; the original evaluator used
8081. The original known-42 result stays in classification.json as a seed
sensitivity check. The separate known-8081 result is needed for exact historical
decomposition; neither is fresh confirmation.

The successful main study used supervised.py SHA256
`a35cc4eb56d5398f87f23386673e314b4c2719680c4274d42b82e01377ba17aa`.
The final source is
`d4ab5f1712036d45f18e14adb8ae040176c193210a40aa4323740266f028ba47`.
A bytewise reconstruction verifies that their only difference is the exception
condition for a failed experimental Extra Trees refit with a completed checkpoint.
The main study had no refit failures. Normal candidate fitting and predictions are
therefore unchanged by this final recovery correction. The harness also corrects
its known-case model-seed parameter. The recorded historical result remains intact.

Next priorities are to validate two holdouts on more development seeds and
mechanism families, address the severe 95% multiclass/interaction failures, retain
accuracy/BA/macro F1 alternatives pending official rules, and review selector gaps
and runtime together before freezing confirmation. Keep production defaults intact.

## Historical loss reproduced with the correct model seed

The separate classification-known8081.json uses dataset seed 8081 and model seed
8081, matching the historical evaluator. It completed in 5.63 seconds. All six
outer candidates were valid and none used fallback. These are previously inspected
rows, so the result diagnoses the known loss; it does not establish generalization.

| Candidate | Fitting weights | Accuracy | Balanced accuracy | Macro F1 | Minority recall |
|---|---|---:|---:|---:|---:|
| Logistic regression | balanced | .83889 | .83086 | .77295 | 81.82% |
| Logistic regression | unweighted | .87778 | .72542 | .76035 | 48.48% |
| Extra Trees | balanced | .85556 | .60606 | .63437 | 21.21% |
| Extra Trees | unweighted | .87222 | .65152 | .69628 | 30.30% |
| Histogram boosting | balanced | .90000 | .79777 | .82000 | 63.64% |
| Histogram boosting | unweighted | .90000 | .76252 | .80392 | 54.55% |

The original change from balanced logistic regression to unweighted Extra Trees
is reproduced exactly: ordinary accuracy rises 3.33 percentage points, while BA
falls 17.93 points, macro F1 falls 7.67 points and minority recall falls 51.52
points. Both weighting and family selection contribute. Along the path that first
removes logistic-regression weighting, BA falls 10.54 points and minority recall
falls 33.33 points. Switching that unweighted model from logistic regression to
Extra Trees then loses another 7.39 BA points and 18.18 minority-recall points.

This allocation is path dependent, rather than a universal causal percentage.
Removing weighting from Extra Trees instead improves BA by 4.55 points. The four
corners show an interaction: balanced weighting has a different effect in each
family. Thus neither weighting alone nor selection alone explains the original
loss, and a blanket instruction to balance or unweight every family is unsupported.

On this known case, the unweighted accuracy selector chooses Extra Trees with
one holdout and logistic regression with two. Two holdouts improve its outer
accuracy by 0.56 points, BA by 7.39 points and macro F1 by 6.41 points. Its BA
still remains below balanced logistic regression. The balanced BA and balanced
macro F1 selectors retain logistic regression with both one and two holdouts.
Histogram boosting has better outer macro F1 here but is not selected by those
policies; retrospective outer superiority never enters selection. This remaining
regret shows why two holdouts should be tested further rather than promoted.

The final classification-only check passed all 12 tests in 7.15 seconds, including
both forced linear and Extra Trees refit failures recovering a completed checkpoint
with different fitting weights. Other relevant pre-study checks passed 28 tests.
The new known-case artifact records the final source hashes separately from the
main study. There were no further production edits, automatic commits or pushes.
