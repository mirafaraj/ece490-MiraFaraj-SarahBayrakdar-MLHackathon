# Second batch: controlled clustering failure analysis

Development protocol declared before running: reproduce all eight first-batch
indicator-removal losses from their exact feature hashes; run indicators on/off
crossed with full-covariance geometry/K-means-only, with two identical repeats.
For these same inputs, change the overall fitting seed twice for indicator-free
geometry; separately change only GMM initialization seeds twice with K-means,
the public fitting seed and the GMM training sample fixed (a harness-only
constructor interception). This separates initialization effects from the
confounded overall seed change. Then test six existing mechanisms with two new development seeds,
12%-of-feature-SD independent noise, and either 30% random missingness or
latent-group-dependent missingness (65% for one designated group per column,
3% for other groups). This is 192 small direct calls, not confirmation or CLI
deadline evidence. Two seeds within six related families are not 24 independent
dataset families.

Latent groups influence missingness only inside the synthetic data generator.
The production model receives features, required group count, fitting seed,
time budget and optional controls; it receives neither reference groups nor
dataset names. All ARI/AMI calculations and per-group missingness summaries
are offline. No representation is selected by reference scores or by comparing
raw BIC values across different feature spaces.

The predeclared mitigation candidate combines the existing indicator-free and
K-means-only controls. It can avoid a harmful GMM solution, but may lose useful
full-covariance structure. All original restarts and BIC comparisons remain in
the geometry control. New per-start GMM diagnostics are observational and use
the existing optional instrumentation flag; default model behavior is unchanged.

The harness preserves historical files and refuses to overwrite its new output.
Evidence contains feature/source hashes, full prediction/reference arrays,
initialization/convergence/component-weight diagnostics, validity and wall time.
Runtime is measured in fixed serial policy order and is descriptive only.

## Results

All **192/192** calls returned aligned predictions with exactly the required
number of groups. No exceptions, empty-group repairs, or emergency outputs
occurred. All **312 fitted GMM candidates report convergence**; the spherical
candidate retains its two internal initializations, whose individual convergence
is not separately exposed. Convergence is not sufficient evidence
of useful groups. Direct-call fitting took 7.375 seconds overall, with a maximum
individual model wall time of 0.219 seconds. These figures exclude interpreter
startup and do not certify the 60-second CLI limit or maximum-size behavior.

All 32 current/no-indicator comparisons against historical prediction hashes
matched exactly, as did their feature hashes and reference metrics. All 32
within-batch identical-repeat prediction-hash pairs also matched. Thus the
historical failures are reproducible; they are not nondeterministic execution
errors. The fresh cells were not identically rerun and represent two development
seeds per mechanism, not independent confirmation.

### Eight exact failures and the factorial intervention

Each cell below is ARI. `Geometry` preserves the existing K-means baseline,
spherical/full covariance fits, generic and baseline-centroid restarts, and
same-space BIC gate. It can retain K-means; it does not force a GMM prediction.

| Mechanism / mask / seed | Indicators + geometry | Indicators + K-means | No indicators + geometry | No indicators + K-means |
| --- | ---: | ---: | ---: | ---: |
| Spherical / random 10% / 6101 | .75408 | .47903 | .74627 | .74627 |
| Correlated / feature 30% / 6101 | .27737 | .27737 | .27347 | .07841 |
| Correlated / feature 30% / 6103 | .30710 | .30710 | .30520 | .27029 |
| Correlated / feature 30% / 6107 | .36978 | .36994 | .35410 | .07327 |
| Curved / feature 30% / 6101 | .15704 | .15704 | .07672 | .07672 |
| Curved / feature 30% / 6103 | .26436 | .26436 | .05482 | .05083 |
| Curved / feature 30% / 6107 | .31872 | .31872 | .08159 | .08159 |
| Missing axis / no extra mask / 6101 | 1.00000 | 1.00000 | .67914 | 1.00000 |

The representation effect, GMM-policy effect and their interaction are separately
saved per case in `experiments/second-batch/clustering-summary.json`. The important
causal distinctions are:

- **Missing-axis exception:** removing indicators does not hurt K-means at all;
  allowing GMM after their removal causes the entire .32086 ARI loss (.24049 AMI).
  The predeclared combined no-indicators/K-means mitigation fixes this case.
- **Curved exceptions:** indicator removal still loses .08031, .21353 and .23712
  ARI with GMM disabled. The two severe losses are therefore not explained by
  GMM alone. For seeds 6103/6107, the K-means AMI losses are .23314/.27994.
  The representation contributes causally under this feature-dependent masking.
- **Correlated exceptions:** removing indicators harms K-means by .19896, .03681
  and .29667 ARI. Allowing GMM recovers .19506, .03491 and .28083 respectively
  within the indicator-free representation. Disabling GMM as a blanket guard
  would destroy this useful recovery.
- **Spherical exception:** removing indicators improves K-means by .26724 ARI,
  but indicators plus geometry already improve on indicators plus K-means by
  .27505. The small final loss comes from this interaction, not a generally worse
  indicator-free K-means representation.

### Is initialization the missing-axis cause?

With input, K-means centers, public fitting seed and GMM training sample fixed,
changing only the GMM initialization seed from 6101 to 6102 or 6103 recovers
ARI/AMI **1.0/1.0**. This isolates initialization sensitivity from the separately
recorded overall-seed experiment, which also changes K-means and sample ordering.
The override is confined to the experiment harness; production seeds are unchanged.

However, the original semantically worse solution has **better BIC: 939.61 versus
1116.33** for the perfect-group solutions, within the same representation. The
original best full-covariance start has weights .196/.333/.471 rather than the
perfect-group solution's .333/.333/.333. All these fits converge. Initialization
changes which solution is found, but density selection favors the worse reference
grouping when that solution is available. Simply adding restarts or checking
convergence would not reliably correct this loss and could retain that solution.
Weight imbalance alone is not grounds for rejection: unequal-density mechanisms
are supported inputs.

GMM-only seed changes alter ARI in four of the eight input cases (six of sixteen
override calls), sometimes making it worse. The severe curved losses for seeds
6103 and 6107 persist under both isolated GMM seed overrides. Mask-alignment ARI
for the missing-axis case rises from .00845 with indicator-free K-means to .14927
with the poor GMM solution; this is consistent with a density/missingness effect,
not a complete explanation of the EM objective. Mask-alignment measurements are
offline diagnostics and are never a model-selection score.

### New parameters and genuinely informative missingness

Each row averages 12 cases: six mechanisms and two new seeds. Wall time is mean
direct-call model time; all rows have 12/12 valid results and zero repairs/errors.

| Missingness | Optional policy | ARI | AMI | Wall seconds |
| --- | --- | ---: | ---: | ---: |
| Random 30% | Current geometry | .14500 | .16913 | .04158 |
| Random 30% | Indicators + K-means | .30367 | .33221 | .02075 |
| Random 30% | No indicators + geometry | .41617 | .44984 | .03650 |
| Random 30% | No indicators + K-means | .45571 | .48431 | .01825 |
| Group informative | Current geometry | .43180 | .44497 | .03508 |
| Group informative | Indicators + K-means | .49232 | .49288 | .01983 |
| Group informative | No indicators + geometry | .37659 | .40881 | .03517 |
| Group informative | No indicators + K-means | .42504 | .46971 | .02083 |

Indicator-free K-means beats current ARI in **all 12 random-mask cases**, by at
least .22353, but loses in **5/12 informative-mask cases**, with a worst loss of
.29398 ARI/.29194 AMI (curved seed7211). Indicator-free geometry improves random
ARI by .27118 on average but loses informative ARI by .05521. Its worst fresh
loss is **.45235 ARI/.44071 AMI** on informative missing-axis seed7201
(.53592/.57812 to .08357/.13741). This materially weakens any proposal for
unconditional indicator removal.

The combined K-means mitigation improves on indicator-free geometry in 9 fresh
cases, loses in 5, and ties in 10; its worst loss against that alternative is
.27220 ARI. It is therefore a useful optional alternative, not a universal fix.

For completeness, family-level mean ARI exposes where aggregate gains come from:

| Mask / family | Current geometry | Indicators + K-means | No indicators + geometry | No indicators + K-means |
| --- | ---: | ---: | ---: | ---: |
| Random / spherical | .13906 | .38482 | .42116 | .45441 |
| Random / correlated | -.00238 | .12199 | .05156 | .23335 |
| Random / unequal density | .16812 | .30748 | .60776 | .60776 |
| Random / mixed | .40035 | .40035 | .71310 | .71310 |
| Random / curved | -.00019 | .10837 | .29554 | .29554 |
| Random / missing axis | .16501 | .49900 | .40792 | .43011 |
| Informative / spherical | .45836 | .45836 | .24722 | .54914 |
| Informative / correlated | .35185 | .41820 | .59567 | .38868 |
| Informative / unequal density | .52167 | .55306 | .50847 | .40708 |
| Informative / mixed | .46370 | .46370 | .52496 | .52496 |
| Informative / curved | .38452 | .58362 | .28983 | .15373 |
| Informative / missing axis | .41068 | .47697 | .09340 | .52667 |

## Retention decisions and next steps

1. **Keep all production defaults and useful full-covariance configurations.**
   No experimental policy passes a universal-safety case. The most favorable
   fresh overall mean belongs to indicator-free K-means (.44038 ARI/.47701 AMI
   versus current .28840/.30705), but this hides the severe informative-mask
   regressions above. Current and alternative policies remain optional controls.
2. **Send the full factorial evidence for independent review.** Indicators-on
   K-means improves mean ARI in both fresh masking scenarios, with a fresh worst
   ARI loss .04390, but the earlier first-batch correlated failure was .52051.
   Preserve that counterexample; this small batch cannot justify replacing GMM.
3. **Develop a training-only stability/representation safeguard before widening
   confirmation.** It must retain useful covariance gains and useful missingness,
   distinguish converged density solutions from reliable group recovery, and
   account for added fitting cost. A convergence check, more restarts, hard-coded
   missingness threshold, dataset-name rule or reference-score veto is not
   supported by these results. No new production selection rule was introduced.
4. **Validate new criteria on new mechanism families and CLI budgets later.**
   The present inspected data become development evidence for any subsequent
   tuning. ARI/AMI are provisional quality scenarios; official scoring/size/memory
   limits and separately approved independent confirmation remain necessary.

The focused run passed **14 tests** (five new and nine existing) in 4.48 seconds:
observational instrumentation, unchanged predictions/selections, ignored supplied
labels/test features, unchanged inputs, full restart counts, deterministic data
generation, exact historical feature hashes, and isolation of GMM-only seeds
with identical training samples and K-means centers. Parent integration tests
remain the final repository-wide check. No evaluator, historical report,
regression model, default configuration, commit or remote was changed by this
clustering investigation.
