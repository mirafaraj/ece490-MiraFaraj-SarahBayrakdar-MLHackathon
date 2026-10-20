# Third targeted batch: primary predictive results

Starting checkpoint: `a49253cbab7579ad407ca0c02d012998b55b918d`.
The only positive screening result is a **smaller-leaf classification alternative
under unweighted fitting**. Larger regression trees fail, and reduced clustering
indicator amplitude has important informative-missingness losses. No production
default or candidate portfolio is changed. Official scoring remains unknown.

## Baseline, current policy and primary candidate

The actual entry-point default is the original baseline. The table distinguishes
it from optional routed policies; improvements do not imply deployed promotion.

| Task / provisional measure | Original baseline | Current optional policy | Candidate |
| --- | ---: | ---: | ---: |
| Classification, BA | .43549 | .58987 | .63811 |
| Classification, macro F1 | .43751 | .62064 | .67272 |
| Classification, accuracy | .94926 | .96037 | .96556 |
| Classification, minority recall | .03571 | .30556 | .38492 |
| Regression, RMSE | 3.47855 | 3.03815 | 3.06650 |
| Regression, MAE | 2.36507 | 1.99234 | 2.03671 |
| Clustering, ARI | .29538 | .25326 | .32341 |
| Clustering, AMI | .29482 | .25671 | .32417 |

Classification compares fixed-unweighted BA selection against the same policy
with histogram minimum leaf5. Both objective policies select identical outputs
on this study. The balanced-BA routed alternative still achieves higher BA
(.70828 current, .71771 leaf5), with lower accuracy (.90185/.90630); retain it.
Regression raw means mix target scales; its gate uses per-case relative effects
with equal family weighting. Clustering compares current covariance geometry
against quarter-amplitude indicators and separately fits the actual baseline.

## Classification: meaningful but narrow progress

All18 cases/288 fits valid, no exceptions/fallback;96.01s offline. The unweighted
candidate gains **4.82 BA points,5.21 F1 points,7.94 minority-recall points**, with
positive seed means on all three seeds. Fixed-histogram BA/F1 gains (.61204 to
.64104 / .65516 to .67141) and lower selector regret support a useful leaf-size
effect, rather than only changing the chosen family. Weak/noisy cases do not
improve. Balanced policies miss their declared primary-metric thresholds.

There are serious counterexamples. Localized binary seed98003 loses12.5 BA
points: minority detections5/8 to3/8. Reused inner bootstrap favors the challenger
200/200 times, showing that tiny-support confidence is not independent evidence.
Balanced-policy losses include23.36 BA points in another case. All severe cases
and class supports are retained. Selected unweighted log loss/Brier improve
(.24718 to .23305 / .07563 to .06607), but fixed-histogram log loss worsens,
so no calibration claim is warranted.

Median unweighted selection work remains.3825s, final histogram work.2735 to.2730s.
This is direct work, not full-process runtime. Targeted independent evaluation
is warranted for the unweighted alternative, explicitly including rare-class
exceptions and wider noisy tables. There is no broad replacement recommendation.

The predeclared wider CLI stress case is a strong limitation, not a hidden win.
All18 calls are valid, no fallback/recovery/implementation failure; maximum
external time12.485s. At25s, baseline/current/challenger median wall times are
5.766/8.922/11.531s; at50s,6.672/9.235/11.579s. Three repeats agree on predictions
and selection at each budget. Baseline and challenger both score accuracy.94667,
BA.33333,F1.32420; current unweighted linear scores.93333/.32864/.32184. **Every
policy has zero minority recall** on the110-feature fixture. The candidate
matches baseline instead of recovering useful minority structure, and costs more
than current. No 60s breach occurred, but this is not deadline-quality progress
or maximum-size certification. Existing balanced routed remains an offline/default-
equivalence comparator; the CLI isolates unweighted policy leaf size only.

## Regression: reject the additional capacity

All144 fits valid, no errors/exclusions;93.23s offline. Capacity31 worsens the
nonlinear/mixed target subset by **1.02% RMSE and2.24% MAE**, negative across all
three seed means. Fixed histogram itself worsens1.97%/3.59%; selections remain
Ridge3/histogram12/Extra Trees3. Retrospective selection regret is zero in the
evaluated portfolios. This batch therefore gives no reason to spend more on
selection folds or final-sized validation for these mechanisms.

Ridge predictions are identical in all three linear controls, preserving17.79%
RMSE and27.63% MAE gains against Extra Trees. The worst selected-policy change
is+4.68% RMSE/+6.56% MAE. P99 residuals improve2.16% mean-relative, but P90 worsens
1.13%, and missing/observed-row MAE both worsen. Tail benefit does not satisfy
the declared error objectives. Overfitting is plausible, not proven: training
residuals were not collected.

Median selection1.125 to1.156s; selected refit.290 to.352s. Keep15 leaves and
the linear/tree/boosting alternatives. No independent score evaluation or CLI
campaign is justified for31 leaves. A future mixed-feature representation study
is better motivated than further capacity or validation spending.

## Clustering: representation compromise remains unsafe

All72 calls valid,108 GMM candidates converged; one empty-group repair per
geometry policy. Quarter amplitude improves overall ARI by.07015 and AMI by
.06745. Random-missingness cases all improve. But informative AMI falls
**.33010 to .31091**, with three severe informative-mask ARI losses (worst.2152;
worst AMI loss.2650). The frozen acceptance gate fails.

Mixed-feature K-means results isolate representation effects: damping helps
irrelevant masks but also dilutes useful masks, without any GMM selection change.
One numeric GMM failure includes a near-zero-weight component and required-k
repair despite convergence; it does not explain every loss. Median direct call
time is.047s for both policies. Convergence and aggregate improvement cannot
justify an unconditional amplitude change. Keep amplitude1 and existing covariance
alternatives; retain.25 only as development evidence for a future training-only
representation/stability safeguard. No extra unsupervised selector was added.

## Verification and evidence limits

- Independent reconstruction passes288 classification records (including probability
  log loss/Brier),216 training-only classification selections,144 regression
  records,72 regression selections and72 clustering ARIs. Full inputs, splits,
  references and predictions are retained.
- **18/18 default prediction/selection equivalence probes pass** against the
  starting checkpoint, including protected regression fixtures.
- All179 tracked files outside three intended edited modules are byte-preserved,
  including baseline models, shared preprocessing, configs, evaluator, historical
  results and portability fixtures. Existing default config values match exactly;
  new controls default to adaptive minimum /15 leaves /amplitude1.
- All18 conditional CLI metrics independently reconstruct from saved references
  and predictions. Regression/clustering failed their gates and were not given
  extra CLI quality campaigns. All repeated CLI predictions and selections match
  exactly, and final source maps still match every completed study.
- **263 repository tests pass**, three expected warnings,124.68s, under Haswell
  BLAS dispatch. Coverage includes optional-control isolation, input/reference
  invariance, training-only selection, lexical labels, portability fixtures,
  checkpoint recovery and spawned deadlines. Test XML is preserved. This local
  Windows run is not a fresh Linux CI run; no push was authorized.
- No full confirmation, secondary model study, commit or push was performed.

All studies are serial and use frozen parameters. Related mechanisms/seeds are
development evidence, not independent unseen-family confirmation. Memory/official
maximum-size, clean-install and official score compatibility remain unverified.
No reference labels enter fitting or production selection. The existing checker,
checkpoint isolation, time reserve and original objective alternatives are retained.

## Next decisions

1. Independently assess the unweighted smaller-leaf classification frontier,
   including severe rare-class losses, probability mistakes and nuisance features.
2. Keep regression15; examine mixed-feature representation rather than enlarging
   the model or adding folds on evidence with zero measured selection regret.
3. Keep clustering amplitude1. A future safeguard must preserve useful masks and
   covariance gains; do not infer a safe missingness rule from these reference scores.
4. Obtain official metrics before selecting task champions. Any later confirmation
   must use frozen candidates and unopened independent families.

Task details: classification.md, regression.md, clustering.md. Raw arrays, source
hashes, summaries, compatibility and verification: `experiments/third-batch/`.
Proposed checkpoint after review:
`checkpoint: evaluate targeted predictive-performance variants`.
Do not commit or push automatically.
