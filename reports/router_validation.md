# Router integration, selector stability, and verification

## Status

The opt-in routed strategy, strict task policies/configuration, completed-baseline checkpoint
recovery, experimental category counts, explicit objectives, tie handling and model card are
implemented. Baseline remains the default. Categorical counts are disabled by default; near-tie
tolerance defaults to zero. Official objectives, checker, practice data and environment limits
are absent, so no official scoring gain or default promotion is claimed.

## Correctness and independent review

Complete integrated suite: **153 passed**, two expected identical-point clustering warnings,
97.04 seconds in the existing Windows Python 3.12.14 environment. Tests cover route determinism,
config rejection, metric/weight alignment, fold-only fitting, lexical/fractional labels, smoothing
and unseen-category priors, missing namespaces, validation directions, baseline parity and reuse,
staged outputs, real worker timeout/error/crash recovery, corrupted checkpoints, fallback provenance,
evaluation-label invariance, supervised test-feature invariance, and row permutation.
After the complete run, fitted-state tests were strengthened to hash retained fitted objects
after prediction, detecting prediction-time mutation as well as pre-fit leakage. All eight
invariance tests passed again in 22.09 seconds. Production source/config stayed frozen.

Independent runtime/policy review is in router_cross_review.md; policy details are in
router_policy.md. Its noted duplicate-baseline refit was subsequently fixed: a completed baseline
is reused when selected and after eligible optional refit failures. Final refit failures are
marked completed-baseline fallbacks rather than silently counted as normal completions.
Optional refit reserves and memory estimates remain heuristics, not hard resource guarantees.

## Offline selector study

Eight supervised families, data seed 42, two outer split seeds 42/314, **148/148 valid direct-call
runs**, stable source/config. Eight-family coverage includes Iris, mostly-zero classification and
regression, missing interactions, heavy-tailed regression, missing products, high-cardinality
classification and mixed/missing regression. These are overlapping splits of eight families,
not 16 independent datasets. Scores are provisional and regression quality is negative loss.

| Selector | Wins / ties / losses against baseline | Selection counts |
|---|---|---|
| Existing auto | 9 / 6 / 1 | Extra Trees 6, linear 7, histogram 3 |
| Routed | 10 / 5 / 1 | Extra Trees 4, linear 9, histogram 3 |
| Same unrestricted portfolio | 10 / 5 / 1 | Extra Trees 4, linear 9, histogram 3 |
| Exploratory near-tie | 10 / 5 / 1 | Extra Trees 4, linear 10, histogram 2 |

These small cases did not trigger the policy's optional-model resource exclusions, so they do
not demonstrate that resource routing itself improves quality. Resource pruning is unit-tested.
Routed's observed loss is high-cardinality split 314: balanced accuracy about 0.03990 below baseline,
with a retrospective gap of the same size to the best evaluated candidate. Per-split gaps,
selection frequencies, validation-to-outer gaps, failures, and timings are recorded in
experiments/selector_stability.json. All candidate-family comparisons were complete; dummy/prior
is included. Retrospective best-evaluated gaps are diagnostics only and never enter selection.
Do not average raw gap magnitudes across incompatible metrics or target scales.

The categorical fixed candidate scored balanced accuracy 0.78008 and 0.73462 in about 0.016/0.014
seconds of direct fit/prediction on its two high-cardinality splits. The full categorical-enabled
selector selected Extra Trees once and linear once; it had one tie/one loss against baseline.
These are limited development results, not confirmation or CLI timing, and support leaving it
disabled by default. Unit fixtures separately verify inner feature/alpha selection and final refit.

The exploratory tolerance was 0.01 classification-score units or 1% of outer-training target standard
deviation for regression, explicitly development-only. It changed sparse-linear classification
split 42 from histogram to linear, reducing balanced accuracy by 0.03889 with roughly 0.066 seconds
less direct-call time. This is not sufficient evidence to adopt a positive tolerance. Zero remains
the configured default, and no positive tolerance was tuned after confirmation inspection.

## Frozen CLI comparison

Completed: nine router families × seeds 42/8081 × baseline/auto/routed = **54/54 valid actual CLI
runs, zero fallbacks**, with stable source/config hashes. Seed 8081 was prospective confirmation
for this frozen implementation only; it is now inspected development evidence for any later
modifications. Iris uses another split of the same
bundled dataset; synthetic families use independently generated rows. Family design was visible
during development. Input hashes match across paired strategies and source/config stability is
checked before/after. Each strategy ran 18 cases; all provisional profile semantic checks passed.

| Seed/partition | Task | Routed wins / ties / losses versus baseline |
|---|---|---|
| 42 / development | Classification (balanced accuracy) | 2 / 1 / 0 |
| 42 / development | Regression (RMSE) | 2 / 1 / 0 |
| 42 / development | Clustering (ARI) | 0 / 2 / 1 |
| 8081 / confirmation | Classification (balanced accuracy) | 1 / 2 / 0 |
| 8081 / confirmation | Regression (RMSE) | 2 / 1 / 0 |
| 8081 / confirmation | Clustering (ARI) | 1 / 1 / 1 |

Against current auto, routed quality ties on all 18 paired cases. Baseline-first checkpointing
and explicit policies improve execution control; this comparison does not show higher model
quality than auto or faster execution. Mean CLI wall times: baseline 4.715 seconds, auto 4.973,
routed 7.127. Routed range: 6.399–9.546 seconds. These are sequential local small-table observations,
not maximum-size/official-environment compliance.

The confirmation imbalanced case illustrates objective uncertainty: balanced accuracy improves
0.6061 to 0.8309 while ordinary accuracy declines 0.8556 to 0.8389. Mean classification confirmation
balanced gain is 7.49 percentage points; ordinary accuracy has one loss/two ties. The missing-axis
clustering case regresses from ARI 0.3838 to 0.2962, whereas correlated axes improves 0.99545 to 1.0.
These observed failures and the density-versus-reference objective gap support retaining baseline.

Reports: experiments/router-comparison.json, router-vs-baseline.json, router-vs-auto.json and
router-vs-baseline-accuracy.json. No rule, threshold, or default was retuned after confirmation.
The categorical candidate and positive tie tolerance were not part of this confirmation and
remain unpromoted. Official objective alignment remains a mandatory default-change gate.

Production/script/config SHA-256 at freeze:
`c7904fae62eb7f4e81b521fff47c83cd511469c5bcc15d25413d7e12ec90e609`.
Exact source snapshot: experiments/router-source-c7904fae62eb7f4e.zip.

## Reproduction

```powershell
.venv312/Scripts/python.exe -m pytest -q
.venv312/Scripts/python.exe -u scripts/selector_stability.py --explore-tie --output experiments/selector_stability.json
.venv312/Scripts/python.exe -u scripts/evaluate.py --suite router --seeds 42 8081 --strategies baseline auto routed --output experiments/router-comparison.json
.venv312/Scripts/python.exe scripts/summarize_comparison.py experiments/router-comparison.json --reference baseline --candidate routed --output experiments/router-vs-baseline.json
.venv312/Scripts/python.exe scripts/summarize_comparison.py experiments/router-comparison.json --reference auto --candidate routed --output experiments/router-vs-auto.json
```

MODEL_CARD.md describes inputs, preparation, metrics, reproducibility, measured versus assumed
limits, and failure behavior. README.md documents --strategy routed and explicit accuracy/categorical
config examples. CPU/thread limits and parent worker deadlines are retained. Offline direct-call
timings are not full CLI timing and have no hard interruption; inference is not burdened with
repeated outer evaluation splits. No network or AI service is needed for prediction.

## Still unverified

Official input/output/profile/metric compatibility, nine official practice datasets, published
maximum sizes and memory limits, clean assessment-environment installation, successful remote CI,
broader independent real-world confirmation, public repository/team metadata and final submitted
commit. Clustering density criteria still do not guarantee semantic group recovery. A checkpoint
protects a completed baseline; a deadline reached before completion can still require weak
emergency output. No individual finite small-table benchmark certifies the 60-second limit.
