# Independent first-batch evaluation — 20 October 2026

**Keep production defaults unchanged. Retain indicator-free clustering and two-holdout
classification as experimental alternatives. Prioritize runtime headroom and the
identified severe losses before expanding search.** No official grading formula is
available; classification accuracy/BA/macro F1, regression RMSE/MAE, and clustering
ARI/AMI remain provisional scenarios, not estimated contest points.

## Confirmed from saved evidence

The independent verifier passed **4,677 consistency checks**. It reconstructed
all 18 classification-policy summaries and their selections from recorded inner
scores, checked selected-candidate outer metrics and retrospective gaps, paired
clustering input hashes, and verified the exact 39+9+6 runtime union. The P0
evaluator was applied to verified actual runtime CLI rows; its family-bootstrap
helper was used descriptively for clustering. Direct-call studies were not
misrepresented as full CLI evidence. All **66 existing P0 tests passed** again.

| Change | Confirmed result | Assessment |
|---|---|---|
| Balanced BA selector, two versus one holdout | Mean accuracy +0.111 pp, BA +0.966 pp, macro F1 +0.682 pp. Nine cases tie on all three metrics; selection changes on only three. Accuracy/F1 have one win and two losses; BA two wins and one loss. | Promising small development effect, not broad or independent evidence. Retain optional. |
| Indicator-free clustering | ARI 0.48195 to 0.74478; AMI 0.51501 to 0.74531. ARI 61 wins/8 losses/21 ties; AMI 63/6/21. Every one of six family means improves. | Strongest modeling candidate in this batch, with unresolved severe exceptions. |
| K-means-only clustering | Mean ARI +0.03128; AMI +0.03038; worst ARI loss 0.52051. | Useful ablation/safeguard; no blanket promotion. |
| Diagonal covariance | Mean ARI -0.03094; AMI -0.02828; worst ARI loss 0.52051. | Reject the claim of a beneficial blanket replacement. Preserve the control, not as the proposed default. |
| Runtime/recovery | 54 unique planned CLI runs, all valid; max external time 27.078s; four completed-baseline recoveries and two emergency outputs. | Execution claims confirmed on these two fixtures only. |

The historical classification decomposition is reproduced at the correct model
seed. Balanced logistic to unweighted logistic loses 10.54 BA points; then moving
to unweighted trees loses another 7.39. This path-dependent decomposition is not
a universal attribution: unweighting trees improves BA on that same case. The
earlier accuracy-focused instruction motivated an experiment, not a production
default change. Neither weighting nor an objective should be imposed globally.

## Significant losses and uncertainty

- **Rare multiclass classification:** case10 loses **39.33 ordinary-accuracy
  points** versus the weighted tree baseline (94.67% to55.33%), while BA improves
  about10.44 points and macro F1 declines about3.50 points. Both one and two
  holdouts choose the same weighted linear model: this failure was not introduced
  by the second holdout. Another case still has zero minority recall. Reported
  outer support of four rows per minority class makes recall highly variable.
- **Indicator-free clustering:** three losses exceed P0's0.10 ARI alert threshold.
  Worst: missing-axis/seed6101/no additional masking, **-0.32086 ARI/-0.24049 AMI**.
  Curved groups under30% feature-dependent missingness lose0.23712 and0.20953 ARI
  on two seeds. All eight ARI losses select full GMM; this is association, not proof
  of a shared cause. Missingness may be useful, and EM/selection basins can change.
- **Deadline-quality regression:** on the wider25-second fixture, routed BA is
  **0.5000 versus auto's0.7562**, with zero minority recall. Routed always recovers
  a weak baseline; auto completes the stronger model twice and uses emergency
  output once. At50 seconds they produce identical prediction hashes in all three
  repeats, but routed averages26.203s versus21.719s. Valid recovery is not evidence
  of good model quality or positive expected score.
- **Runtime attribution:** checkpoint writes are tiny; parent import/configuration
  work is a plausible overhead source. The entire2.4s residual has not been
  causally attributed to imports, and removing it alone may still leave too little
  time for the wider optional refit.
- **Generalization:** twelve rotating classification cases share one seed and
  related generators. Clustering has six related mechanisms, not90 independent
  datasets. Exploratory six-family bootstrap intervals for indicator removal are
  positive (ARI improvement0.190–0.342; AMI0.160–0.301), but do not overcome P0's
  eight-family requirement or establish informative-missingness safety. All
  inspected seeds are development evidence for future changes.
- **Evidence gaps:** classification saves metric/prediction hashes but not per-case
  input hashes, full prediction/reference arrays or complete repeated-run records.
  Selection and aggregate arithmetic are independently verified; raw classification
  metrics are not independently recomputed. Memory samples are all null. There is
  no official maximum-size, clean-install or assessment-environment certification.
- **Regression:** no new regression-quality study was performed. Eighteen saved
  default-equivalence records match, including regression fixtures, but this is
  compatibility evidence, not new generalization or RMSE/MAE superiority.

## Next batch: controlled changes and acceptance criteria

These are development experiments, not an authorization for a confirmation campaign.

1. **Runtime headroom first.** Isolate parent imports/config loading with phase
   timing, then compare lightweight/lazy loading against current routed with auto
   as an anchor. Retain worker isolation and checkpoints. Use the same two shapes,
   randomized serial order and10/25/50-second budgets; three paired repetitions,
   extending only the ambiguous wider25-second cell to five. For an import-only
   change, accept as an engineering improvement if generous-budget prediction
   hashes/selections remain identical and paired median external time falls by
   at least1s. Report tight-budget completion quality separately; do not claim a
   deadline-quality improvement unless it is measured. If refits still miss,
   separately test admission using measured validation-fit cost and final-fit
   headroom, preserving histogram's50-second gains. Reject invalid output, broken
   recovery, external60s breaches or unreported quality losses. Speed gains alone
   do not justify removing useful models/safeguards.
2. **Resolve indicator-free failures before representation routing.** Reproduce
   the eight losses, then run a fixed2x2 ablation: indicators on/off and
   full-covariance geometry/K-means-only, keeping restarts and all other controls
   fixed. Add genuinely group-informative missingness and new parameter settings
   across the existing six mechanisms. Report ARI and AMI by mechanism, missingness
   and seed. Advance an alternative only if gains persist across both scenarios
   and severe losses have a reproduced explanation or mitigation; promotion needs
   new independent family evidence and P0 uncertainty checks. Do not compare raw
   BIC across different transformed feature spaces as evidence for representation
   selection, or hard-route using dataset names/reference scores. Extra fitting
   and unsupervised-selection cost must be measured before deployment.
3. **Rare-class selection and two-holdout replication.** Keep the three-family
   portfolio; do not expand both weighting variants globally. Repeat the12-case
   design on three new development seeds, record each fold's class supports,
   input/split hashes and predictions, and compare one/two holdouts under an equal
   total selection cap. Add separate rare-multiclass interaction mechanisms; log
   completed/excluded candidates to expose order/budget bias. Require replicated
   benefit beyond the original three changed cases and disclose severe losses,
   selection frequencies and retrospective gaps under all three classification
   scenarios. Retain conflicting policies rather than impose an accuracy or BA
   veto. Do not promote from overlapping splits or tune on outer best-candidate
   labels. More validation consumes refit headroom; test that cost explicitly.
4. **Protect regression and fill external gaps.** Preserve existing candidates,
   run compatibility checks after runtime changes, and reserve a later bounded
   selector study for fresh nonlinear/missing-product/heavy-tail families with
   both RMSE and MAE. Obtain written scoring/checker rules and official size/memory
   limits before selecting task champions. Then plan clean installation, resource
   verification and separately approved unopened validation. No score increase
   can be quantified from the present provisional scenarios.

## Reproducibility and changes made by this review

The classification study's one documented source difference was independently
reconstructed byte-for-byte: removing the final tree-refit exception guard recovers
its recorded hash. Runtime production/config maps match all three segments and
current files; clustering source hashes also match. Integration XML records153
passing tests with no errors/failures; the separate reported35-test run was not
rerun here. Default configuration, baseline dispatch and shared preprocessing have
no diff against the earlier checkpoint.

Only three evaluator-owned files were added in this directory: `verify.py`,
`evidence.json`, and this report. Existing evaluator code, experiment controls,
production files and historical reports were not edited. No fits, dependencies,
commits or pushes were performed. The starting HEAD for this review is
`4885559f0f7bbea6a68cbfe85319fdd5fbebb8a3`; the development report's older starting
checkpoint is historical context, not the current HEAD.

Final preservation check: all139 previously tracked files remain byte-for-byte
unchanged, HEAD is unchanged, and Git reports only this new review directory.
