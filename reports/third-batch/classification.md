# Classification primary study: frozen protocol

Protocol frozen before fits. Seeds 98003, 98017, 98029; 18 tables: linear control, localized nonlinear minority interaction, weak/noisy minority signal × binary/three-class outcomes × three seeds. Each contains 600 rows, six Gaussian numeric and two categorical columns. Majority support is exactly 570; binary minority 30, multiclass minorities 15 each. Labels are top-30 score ranks; multiclass subdivides by a separate numeric axis. Numeric entries independently have 3% missingness. Exact formulas are versioned in the study script. These synthetic mechanisms test generalization across fresh seeds, not arbitrary real-world coverage.

Identical stratified outer 75/25 and inner 75/25 splits use the case seed. Training-only preprocessing. Eight configurations: Extra Trees, linear, original histogram and histogram leaf-5 under balanced/unweighted class weights. Each fits once on inner training (validation-sized models) and once on complete outer training (final-sized models): 288 fits. Leaf-5 changes only the histogram minimum leaf count; existing histogram leaves 15, iterations 60/100 and all other parameters remain fixed. No thresholds/calibration or additional folds. Probabilities come directly from the same underlying `_fit` used by the public candidate API.

Selection policies compare original versus replacement three-family pools, separately for each weight and accuracy/balanced-accuracy/macro-F1 objective. Inner predictions alone determine decisions, with existing linear/trees/histogram tie preference and descriptive 20-second shared cap (50-second model budget × 0.4). The cap was corrected before any fits or results. Offline expanded fitting is not prediction-time budget evidence. All outer predictions, probabilities, labels, supports, inputs, split indices, hashes, time, exclusions and retrospective gaps are retained. Two hundred stratified resamples of saved inner predictions quantify score uncertainty without new model fits; intervals do not constitute independent validation.

Screening gate: at least one percentage point improvement in declared balanced accuracy or macro F1, improvement on at least two seeds, and fixed-candidate benefit or reduced selection regret. Every ≥10-point deterioration is reviewed. Conflicting accuracy alternatives remain available. No default promotion, official-metric assumption or confirmation campaign. Runtime/validity/fallback CLI verification is coordinated separately by the parent.

## Results

All 18 cases / 288 fits completed without exceptions or fallback in 96.01 seconds. Four focused tests passed. No candidate was excluded by the descriptive cap. These are small offline fits, not evidence that every legal dataset completes within 60 seconds; bounded CLI checks are reported in the batch summary.

Means below compare the same 18 outer sets. BA is balanced accuracy. Every policy retains the same three estimator families; leaf-5 replaces the histogram configuration rather than expanding its pool.

| Fixed weighting / selection objective | Accuracy current→leaf-5 | BA current→leaf-5 | Macro F1 current→leaf-5 | Minority recall current→leaf-5 |
|---|---:|---:|---:|---:|
| Balanced / accuracy | .95630→.95778 | .58733→.60806 | .58474→.60903 | .31647→.34623 |
| Balanced / BA | .90185→.90630 | .70828→.71771 | .65289→.66834 | .55655→.57242 |
| Balanced / macro F1 | .95407→.95852 | .68848→.66990 | .66492→.66735 | .48313→.45040 |
| Unweighted / accuracy | .96185→.96519 | .60197→.65009 | .63094→.67418 | .32937→.40774 |
| Unweighted / BA or macro F1 | .96037→.96556 | .58987→.63811 | .62064→.67272 | .30556→.38492 |

The actual balanced Extra Trees baseline averaged accuracy .94926, BA .43549, macro F1 .43751, minority recall .03571, log loss .35450 and Brier .08712. A no-fit training-prior dummy averaged accuracy .94778, BA .41667, macro F1 .40554, minority recall zero, log loss .22363 and Brier .09972. This high accuracy/very low recall is why objective alternatives must remain explicit. The routed balanced-BA policy still attains greater average BA than the unweighted alternatives, despite a large accuracy cost; leaf-5 does not settle the official-metric question. The bounded shortlisted CLI comparison uses the same explicit **unweighted** control on both current and leaf-5 policies to isolate leaf size; that comparator is not the balanced routed default.

**Screening:** only the fixed-unweighted BA and macro-F1 replacement policies pass the preregistered ≥1 percentage point / ≥2-seed gain criterion. BA improves 4.82 points, F1 5.21 points, and minority recall 7.94 points. Both declared metrics improve on all three seed means: BA +.08006/+.05038/+.01428 and F1 +.09828/+.04511/+.01287. Mean retrospective BA regret falls .05690→.02482; F1 regret .06989→.02048. The balanced-BA policy's .94-point BA improvement narrowly misses the gate, and balanced-F1 improvement is only .24 points. No result-dependent retuning was performed.

Mechanism evidence is partial but real: fixed unweighted histogram BA improves .61204→.64104, F1 .65516→.67141 and minority recall .34127→.39286. Unweighted-BA selector gains occur on linear controls (.70968→.77155 BA) and localized interactions (.64387→.72670), while weak/noisy cases are unchanged (.41608). Thus the gain is not merely a selector artifact, and it does not invent useful signal in the noise control. Fixed balanced histogram BA is essentially unchanged (.64572→.64547), confirming that the leaf-size effect interacts with weighting. Smaller leaves are not a universal minority-recall remedy.

Probability results also conflict. For unweighted BA selection, log loss improves .24718→.23305 and multiclass Brier improves .07563→.06607. Yet the fixed unweighted histogram's log loss deteriorates .27485→.29433 while Brier improves .07153→.06789, consistent with some confident mistakes. Balanced-BA selection log loss deteriorates .38099→.39049 while Brier improves .14904→.14412. No probability calibration claim is justified.

## Severe regressions and uncertainty

Four distinct case IDs account for the ≥10-point minority-recall or BA/F1 policy losses; all are retained in `classification-summary.json`.

* Localized binary / 98003, unweighted: original histogram correctly identifies 5/8 outer minority examples, leaf-5 identifies 3/8; BA falls .81250→.68750 and F1 .87939→.76408. Inner minority support is six and its BA increases .66667→.83333, so the training-only selector cannot detect this outer reversal. Inner bootstrap chooses leaf-5 in 200/200 resamples: certainty on a tiny reused validation set does not prove generalization. This unresolved exception prevents a broad replacement recommendation.
* Linear binary / 98029: balanced leaf-5 changes minority recall 7/8→4/8, BA .93750→.75000 and F1 .96491→.82639 despite identical inner BA. Unweighted leaf-5 changes minority recall 6/8→5/8. This is a fixed-model generalization reversal, not just a weighting/selection change.
* Linear multiclass / 98003, balanced BA/F1: inner evidence switches weighted linear→leaf-5 histogram, but outer BA falls .89554→.66197 and mean minority recall .875→.500. Each inner rare class has only three rows, versus four in the outer set. Selection uncertainty remains consequential.
* Linear multiclass / 98017, balanced accuracy: losing the original histogram to balanced Extra Trees changes BA .58333→.33333 and F1 .66204→.32420 while ordinary accuracy changes little. This shows that accuracy-focused selection can still abandon minority information. The balanced-F1 policy's leaf-5 fit also loses .125 mean minority recall.

Median train-only selection time is .852→.836 seconds for balanced policies and .3825→.3825 for unweighted policies. Histogram final-fit median is .6875→.6875 seconds balanced and .2735→.2730 unweighted. There is no meaningful observed runtime penalty on these small tables. All probability vectors, per-class supports and recalls, model-selection frequencies, bootstrap intervals/frequencies, exclusions, input hashes and split indices are preserved. No additional folds or calibration were introduced.

**Recommendation:** the optional unweighted leaf-5 alternative warrants targeted independent evaluation, including the severe binary interaction loss and probability overconfidence; preserve the original histogram, weighted alternatives and all objective-specific policies. It passes a development screening gate, not a universal-improvement or promotion gate. Eighteen synthetic tables, three seeds and very small rare-class validation supports cannot establish expected real-world hackathon gains. Production defaults remain unchanged; no evaluation labels enter selection.
