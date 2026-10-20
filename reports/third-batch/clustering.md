# Reduced missing-indicator amplitude: frozen development protocol

Protocol frozen before fitting. No reference labels enter production fitting or selection.

- 240 rows per case, families correlated/curved/anisotropic with irrelevant masked features/mixed numeric-category groups; seeds 99013 and 99041; masks none/random 30%/group-informative 65% versus 3%.
- Explicit scalar Gaussian transforms avoid multivariate-normal LAPACK roots. Correlated groups use correlation .97. Curved groups use make_moons noise .08. Anisotropic groups have four irrelevant standard-normal axes, each intrinsically 70% missing; `none` means no additional masking in this family. Mixed groups have weak numeric centers and a category agreeing with latent group 80% before replacement (replacement may coincide). Categories are not masked; numeric-only amplitude does not alter encoding.
- Compare unchanged geometry amplitude 1 against .25: **48 direct geometry calls**. Fit actual production baseline separately for every case: **24 baseline calls**, **72 total calls**. No assumption that geometry-selected output equals baseline. Each call has a 20-second internal model budget, not an external CLI guarantee.
- Preserve all covariance families, starts, restarts, BIC margins, numeric value scaling and categorical encoding. Amplitude changes only numeric missing indicators after existing standardization. Default 1 follows the identical matrix path.
- Store full inputs, masks, references, predictions, hashes, selected models, BIC, GMM convergence, timings and repairs. Hashes identify saved inputs, not cross-platform regenerated goldens. No retrospective parameter tuning.
- Report ARI/AMI by family and missingness, worst losses, validity and runtime. Acceptance: mean ARI gain at least .02, positive AMI gain, nonnegative mean ARI/AMI changes for both random and informative masks; explain any new loss of .10 or more before recommending a broad change. Existing eight historical losses remain regression sentinels in the unchanged portable historical-input tests, not fresh success evidence.
- Run correctness, input immutability, reference/test-feature invariance and default-equivalence checks. Parent coordinates serial CPU study and end-to-end budget checks. No stability-veto follow-up in this batch.

## Measured results

All 72 calls completed with valid length and requested occupied groups. The 48 geometry calls fitted 108 GMM candidates; all converged. Each geometry policy repaired one empty group; the actual baseline repaired none. Total direct-call experiment time was 4.062 seconds, excluding interpreter startup. No exceptions or fallback occurred. These small calls do not establish maximum-size CLI deadline compliance.

| Case set | Actual baseline ARI / AMI | Current geometry ARI / AMI | Amplitude .25 ARI / AMI |
|---|---:|---:|---:|
| All 24 | .29538 / .29482 | .25326 / .25671 | **.32341 / .32417** |
| No additional masks (8) | .32902 / .31706 | .45065 / .42186 | .48573 / .46922 |
| Random masks (8) | .08020 / .08172 | .02052 / .01818 | **.19142 / .19237** |
| Informative masks (8) | **.47692 / .48568** | .28861 / .33010 | .29308 / .31091 |

The quarter-amplitude variant gains mean ARI .07015 and AMI .06745 against current geometry, and .02803/.02935 against the actual default baseline. Both seeds improve mean ARI against current geometry (+.09154 and +.04876), though these are only two seeds from four fixed mechanisms. Twelve pairs improve ARI, four deteriorate, and eight tie. All eight random-mask pairs improve ARI/AMI.

The frozen acceptance rule **fails**: informative-mask mean AMI falls .01919, despite its mean ARI rising .00447. Three informative-mask pairs have a loss of at least .10:

| Family / seed | ARI change | AMI change | Evidence and explanation |
|---|---:|---:|---|
| Anisotropic noise / 99041 | −.11011 | −.11510 | Quarter-amplitude selected full GMM has a component weight approximately 9e-18 and requires an empty-group repair. All starts converged; convergence alone does not detect a useful partition. Representation and changed initialization basin remain intertwined. |
| Mixed / 99013 | −.10711 | −.18599 | Both policies use K-means, so this loss cannot be attributed to GMM. Reducing deliberately informative masks dilutes useful grouping information. |
| Mixed / 99041 | −.21520 | −.26502 | Both policies use K-means; same demonstrated representation risk. |

Numeric value columns and categories are byte-identical between representations, as tested. Reduced amplitude directly improves mixed/random K-means grouping; this supports a geometric missing-indicator mechanism without a GMM-selection confound. For numeric GMM cases, scaling changes initialization and regularization behavior as well as representation, so we do not claim a single established cause for every gain. BIC is never compared across representations. Both policies select the same model family in every paired case (16 full-GMM and 8 K-means selections each).

Current and quarter-amplitude median direct-call wall time are both .047 seconds; maxima .079 and .093 seconds. Actual-baseline median is .031 seconds (first-call maximum .219 seconds). Timing is too small and coarse to claim a speed gain. No candidate portfolio, covariance, restart or selection-budget change was made.

## Correctness and decision

Focused verification: **22 tests passed** (16 new plus 6 existing second-batch clustering tests), including sparse/dense numeric-only scaling, categorical preservation, no-indicator/constant-only paths, invalid controls, default-amplitude prediction and selection invariance, reference-label/test-feature invariance, input immutability, deterministic protocol generation and exact historical-fixture replay. The historical fixtures and reports were not modified.

Saved evidence: `experiments/third-batch/clustering.json` contains full inputs, masks, references, predictions and source hashes; `clustering-summary.json` contains all 24 paired changes and group summaries. Reproduce with `PYTHONPATH=src .venv312/Scripts/python.exe scripts/third_clustering.py --output <new-path>`; outputs are never overwritten. An initial invocation without PYTHONPATH failed before imports or fitting, then the declared study completed with PYTHONPATH set. No fits were repeated or parameters changed after results.

**Decision:** retain the optional amplitude control for research, default 1 unchanged. Do not recommend .25 as a general independent-evaluation candidate under the frozen gate. It offers meaningful evidence of improvement under random missingness, but weakens useful missingness in other cases and still trails actual baseline markedly on informative masks. Any future training-only safeguard must first show it can preserve informative structure and avoid collapsed GMM partitions; no safeguard or stability veto was implemented or tested here. Parent owns end-to-end CLI deadlines, overall default-equivalence and final integration verification.
