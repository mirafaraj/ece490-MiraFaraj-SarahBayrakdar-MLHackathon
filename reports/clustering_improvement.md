# Targeted clustering restart improvement

Keep the baseline default. The optional clustering selector now compares a generic
full-covariance initialization with a start using the completed K-means centers.
It spends the same total two full-covariance EM restarts, retaining the converged
solution with lower BIC. The spherical comparison and its conservative BIC margin,
sample/feature/group bounds, labels API, mixed-feature route, and outer deadline
are unchanged. No reference labels enter production selection.

## Diagnosis and development

The observed noisy_blobs seed-42 regression was reproduced: baseline ARI 1.0000,
previous auto 0.633748. Permuting the same full training sample changed GMM's
initialization and sent its generic starts to an inferior density optimum.
Starting one candidate at the already fitted baseline centers recovers ARI 1.0000
and a better BIC. This is a local-optimization mitigation, not a semantic-quality
guarantee or a requirement that the final groups agree with K-means.

The broader development check used eight families at observed seeds 42/314:
noisy blobs, anisotropic blobs, moons, mixed missing/categorical clusters,
spherical blobs, unequal group sizes/variances, circles, and correlated ellipses.
Sixteen direct model comparisons show four wins/twelve ties/no losses versus
baseline. Relative to previous auto, there are two wins/fourteen ties/no losses:
noisy42 recovers 0.633748 to 1.000000; ellipse314 improves 0.494207 to 0.562207.
Anisotropic42 retains 0.831472 to 1.000000 versus baseline, and ellipse42 retains
0.499714 to 1.000000. The previous implementation for additional development
comparisons came from the preserved source-81a8a622c85eca5d.zip snapshot, executed
in memory. No historical report was changed.

## Frozen fresh-seed confirmation

Seeds **2027 and 4099** were declared before evaluation. Clustering source and
tests were frozen before generating these rows; there was no post-confirmation
tuning. Five families, two seeds, two strategies produced **20 valid actual CLI
runs, zero fallbacks**. Every baseline/auto pair has the same input hash. Owned
source/test/harness fingerprints stayed unchanged throughout the run.

| Seed | Family | Baseline ARI | Improved auto ARI |
|---|---|---:|---:|
| 2027 | Noisy blobs | 0.976092 | 0.991932 |
| 2027 | Anisotropic | 0.487975 | 0.501893 |
| 2027 | Moons | 0.493483 | 0.493483 |
| 2027 | Mixed clusters | 0.489879 | 0.489879 |
| 2027 | Ellipses | 0.499845 | 0.498810 |
| 4099 | Noisy blobs | 0.836778 | 0.836778 |
| 4099 | Anisotropic | 0.899482 | 0.986415 |
| 4099 | Moons | 0.468211 | 0.468211 |
| 4099 | Mixed clusters | 0.431568 | 0.431568 |
| 4099 | Ellipses | 0.496532 | 1.000000 |

Against baseline: **four wins, five ties, one loss**. Ellipse2027 loses 0.001034
ARI; its geometry remains difficult. Confirmation compares improved auto with
baseline, not with the previous auto implementation. Families were visible during
development; new seeds supply new samples, not new real-world families.

Observed CLI times were **5.84–7.35 seconds** with a 50-second model/run budget and
60-second harness timeout. Some first cases overlapped an automated-test run;
these are operational observations, not uncontended speed comparisons or evidence
for maximum-size runtime compliance.

## Verification and reproduction

Focused tests: **13 passed**, one expected constant-point convergence warning.
They cover both initialization routes, recovery of a noisy numeric fixture,
legitimate changes to K-means geometry, failed/nonconverged candidates, label
independence, mixed data, low budgets, and required group occupancy.

Final complete suite after confirmation: **68 passed**, two expected identical-point
convergence warnings, 31.91 seconds. No production or test edits followed the
fresh-seed freeze.

Commands (from the project root):

```powershell
.venv312/Scripts/python.exe -m pytest tests/test_clustering_candidates.py -q --basetemp experiments/clustering-improvement-pytest
.venv312/Scripts/python.exe experiments/clustering-improvement-evaluate.py
.venv312/Scripts/python.exe -m pytest -q --basetemp experiments/clustering-improvement-final-pytest
```

Evidence: experiments/clustering-improvement-before.json (original broad cases),
clustering-improvement-development.json (sixteen direct comparisons), and
clustering-improvement-confirmation.json (twenty CLI runs, profiles, timings,
input hashes, source fingerprints). The confirmation harness is preserved as
experiments/clustering-improvement-evaluate.py.

Frozen SHA-256 fingerprints:

- clustering.py: af71fae9e9b9d3bf9f9130474ba438bfed0c3db8dbfc5ca87c359af9d04f25c7
- test_clustering_candidates.py: a94008634f289d474bdc6a97aeba7e640c0aec4c8ee4415c92e3f2db360f7f79
- confirmation harness: ba769dd7d6e5dc6192f35184d2692830f637afa4bc73ebe14d60e6022ad82815

## Remaining limits

The targeted observed regression is fixed on development; general semantic
clustering superiority is not established. Density BIC can still disagree with
reference groups. Moons/circles remain weak, mixed tables retain baseline,
sampling can miss rare groups, and optional fits still rely on the external worker
for hard interruption. The changed sequence checks the time between each
full-covariance restart; insufficient remaining time may skip the second start.
Maximum-size/memory tests and official contract/checker/metrics remain outstanding.
All figures are local development metrics, not contest scores.
