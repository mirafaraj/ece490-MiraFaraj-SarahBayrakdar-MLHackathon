# What we learned on real datasets

**The checks are complete: 147 benchmark CLI calls, 108 on real datasets and 39
stress calls.** All 108 real-data calls were valid and used real models. Stress
testing found two invalid calls and seven emergency outputs. Nothing was committed
or pushed, and no existing project file was changed by this task.

## 1. How you actually run the program

This command by itself does not run a dataset:

```powershell
.\.venv312\Scripts\python.exe main.py
```

It exits with code2 because `--input` and `--output` are required. Once those are
supplied, leaving out `--strategy` means **baseline**, not auto or routed.
`src/unknown_dataset/runner.py:89` contains:

```python
parser.add_argument("--strategy", choices=("baseline", "auto", "routed"), default="baseline")
```

[FACTS.md](FACTS.md) lists the file/line evidence for every input/output rule.
The README and code agree on the basic files and formats. Its blanket statement
that model errors always return nonzero needs qualification: routed may return
valid recovered baseline output after an optional failure. Some README status
statements are historical. The official organizer contract is still unavailable,
so this verifies the repository's **provisional** instructions only.

## 2. What was compared

- Classification: breast_cancer, wine, digits, iris, OpenML mammography and blood
  transfusion. The latter datasets have minority shares2.32% and23.80%.
- Regression: diabetes, California housing and House Prices. House Prices has80
  input columns, including43 categorical columns. All downloads succeeded; the
  initially prepared breast-cancer fallback variants were **not benchmarked**.
- Clustering: iris, wine and digits, with k=3/3/10. All feature rows are clustered;
  the program never receives their reference labels.
- Three seeds42/314/2027. Classification uses stratified75/25 splits; regression
  uses75/25 splits. Every strategy receives exactly the same input per seed.
- Every prediction comes from `main.py` in a fresh process, with default
  configurations,50s internal budget and a60s external guard. No lazy-import or
  third-batch experimental flag is enabled. Held-out labels live separately in
  `references/`, not inside the CLI input directories.

For all36 real dataset/seed comparisons, **auto and routed prediction files are
identical**, not merely equal in their rounded scores.

Metric guide: accuracy is the share correct; balanced accuracy gives each class
equal importance; macro F1 balances false alarms and missed examples separately
for every class. Minority recall counts how many rare examples were found.
RMSE/MAE measure numerical error (lower is better); RMSE penalizes large mistakes
more. R2 is better near1 and a mean-only predictor is near0. ARI/AMI measure
agreement of groups with reference labels after adjusting for chance (higher is
better). Classification scores below are displayed as percentages; CSV scores
use the usual0-to-1 scale.

## 3. Classification: there is no single winner

Means give each of the six datasets equal weight, then average its three seeds.
Percentage-point change means subtraction, not relative percentage improvement.

| Metric | Baseline | Auto | Routed | Auto/routed change vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Accuracy |92.23%|90.66%|90.66%|−1.57 points|
| Balanced accuracy |86.24%|90.14%|90.14%|+3.90 points|
| Macro F1 |87.23%|85.78%|85.78%|−1.45 points|
| Minority recall |76.52%|88.29%|88.29%|+11.78 points|
| Mean complete process time |6.04s|7.32s|10.38s|Auto faster than routed|

**Evidence:** baseline wins average accuracy and macro F1. Auto/routed find more
minority examples and win balanced accuracy. The official metric would determine
which tradeoff matters. Minority recall here means mean recall of the least
frequent training class(es); ties are averaged. Its interpretation is less useful
on a naturally balanced table such as iris.

**The losses are substantial in some cases:**

- **Digits:** auto/routed lose accuracy, balanced accuracy and macro F1 on all
  three seeds. Worst balanced-accuracy loss1.58 points; worst minority-recall
  loss6.98 points. Mean balanced-accuracy loss across seeds is1.26 points.
- **Mammography:** accuracy and macro F1 worsen on every seed. Worst accuracy
  loss8.51 points; worst macro-F1 loss19.83 points. Across seeds the losses are
  5.89 accuracy points and12.81 macro-F1 points on average, despite better recall.
- **Blood transfusion:** accuracy falls on seeds314/2027, by3.74/4.28 points.
  Other metrics can improve at the same time.
- No real-data metric loss versus baseline was observed on breast_cancer, wine
  or iris in this comparison. Ties and gains remain in the full tables.

An example makes the mammography tradeoff concrete. On seed42, baseline finds
34 of65 minority examples and incorrectly flags6 majority examples. Auto/routed
find56 of65, but incorrectly flag258 majority examples. Better recall is real;
so is the extra false-positive burden.

**Explanation and guess:** the confirmed selection objective is balanced accuracy,
not ordinary accuracy or macro F1. All these standard classification policies use
balanced weights; this check did not isolate weighting effects. On digits, the
selector chose histogram on two seeds and linear on one, while the baseline
Extra Trees generalized better. A single validation split and smaller validation
models can rank families differently from the final test; that explanation is
plausible, not causally proven here.

**Recommendation:** retain baseline for an accuracy/macro-F1 goal; consider auto
for a balanced-accuracy/minority-recall goal. Do not change a default before the
official objective is known. The large-shape failures below prevent a universal
recommendation for any classification strategy.

## 4. Regression: auto gives the best observed quality/time tradeoff

Different datasets have different target units. Averaging their raw RMSE values
would mostly measure the dollar scale of House Prices. Instead, the table averages
within-dataset percentage error reductions against baseline; positive means better.

| Dataset | Auto/routed RMSE reduction | Auto/routed MAE reduction |
| --- | ---: | ---: |
| Diabetes |2.14%|3.76%|
| California housing |2.94%|−1.38%|
| House Prices |1.87%|5.03%|
| Equal-dataset mean |**2.32%**|**2.47%**|

Mean R2 changes.71082→.72453. Mean full-process time is12.58s baseline,
10.09s auto and18.30s routed. Auto/routed have identical predictive quality.

**Losses, not hidden by the means:** diabetes seed42 RMSE worsens3.07%
(51.73→53.32), and R2 falls.51610→.48590. California MAE worsens on every
seed; the largest increase is2.26% on seed2027. House Prices has no measured
RMSE/MAE/R2 loss versus baseline; seed2027 ties after selecting Extra Trees.

**Explanation and guess:** selection optimizes RMSE, so better RMSE with worse
MAE on California is a genuine objective conflict. Diabetes selected Ridge, which
lost on seed42 but won on the other seeds. Sampling uncertainty or a modest
nonlinear advantage for baseline are plausible explanations; no causal ablation
was run. Histogram won selection for California and two House Prices seeds.

**Recommendation:** auto is the better observed choice for these ordinary-size
regression tables, with modest gains and less time than routed. Keep baseline as
an alternative, and do not assume a2.32% gain on every unseen table. Both large
regression stress shapes failed to finish a real model under every strategy.

## 5. Clustering: auto wins here, but the gain is concentrated in iris

| Metric | Baseline | Auto | Routed | Change vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Mean ARI |.68233|.77802|.77802|+.09570|
| Mean AMI |.73108|.81293|.81293|+.08185|
| Mean full-process time |5.38s|5.37s|7.74s|Routed is slower|

**Evidence:** all gains come from iris, where auto/routed select full-covariance
Gaussian mixtures. Wine and digits tie baseline on every seed and keep K-means.
No ARI/AMI loss versus baseline occurs on these three datasets. The worst dataset
mean change is therefore zero, not a hidden loss.

**Explanation and guess:** the mixture's geometry gives a partition closer to
iris's reference classes. That does not prove density/BIC selection always finds
the groups we want; no broader grouping or missingness claim follows from iris.

**Recommendation:** auto has the best observed quality/time tradeoff in this
small real clustering set. Preserve baseline and do not extrapolate the gain to
all unfamiliar cluster structures.

## 6. Stress tests: time and valid output are different from model success

Times below are full-process seconds. **Emergency** means valid fallback output,
not successful training. **FAIL** means invalid/missing output. Ordinary pass
means a real model returned valid output before60s.

| Case / task | Baseline | Auto | Routed |
| --- | --- | --- | --- |
|100k×50 classification|50.90 FAIL|27.33 pass|54.01 FAIL|
|100k×50 regression|47.33 emergency|47.41 emergency|47.58 emergency|
|100k×50 clustering|9.35 pass|9.11 pass|12.94 pass|
|5k×500 classification|13.55 pass|47.56 emergency|23.99 pass|
|5k×500 regression|47.37 emergency|47.38 emergency|47.52 emergency|
|5k×500 clustering|8.15 pass|8.02 pass|12.07 pass|
|About40% missing / classification|5.77 pass|7.46 pass|10.00 pass|
|5,000 training category levels / classification|7.15 pass|8.31 pass|11.31 pass|
|Constant/duplicate columns / clustering|5.29 pass|6.06 pass|7.02 pass|
|50 training rows / classification|5.26 pass|5.29 pass|8.00 pass|
|25 classes / classification|5.55 pass|15.33 pass|18.30 pass|
|Numeric-looking labels / classification|5.28 pass|6.67 pass|9.66 pass|
|All-missing column / regression|5.60 pass|5.47 pass|8.45 pass|
|**Maximum observed time**|**50.90s**|**47.56s**|**54.01s**|

All nine calls over45s appear in the first, second, fourth and fifth rows above:
two invalid classification calls and seven emergency calls. No external60s timeout
or unhandled process crash occurred. No real-data run exceeded45s; its maximum
was36.53s. No completed-baseline checkpoint recovery was observed.

The two failures return code1 and exactly:
`ERROR: Output serialization exceeded the execution budget`.
The CLI's own50s budget check rejected and removed its result files, even though
external time stayed below60s. There is no surviving model metadata for those
calls, so their attempted fallback/selection state is **unknown**. A blank/none
field in the raw stress table is not evidence that no fallback was attempted.

The six large regression calls return training-mean predictions, with R2 near0.
Auto's wide classification fallback returns a majority class. These pass output
and time checks while failing to deliver a trained predictive model. For emergency
calls, the table names the final output model; the pre-interruption candidate is
not observable in the saved metadata.

Routed's wide classification trace excludes histogram because its estimated dense
allocation exceeds limits and completes linear. Auto instead hits its worker
deadline; the precise interrupted candidate is not recorded. Resource filtering
is a plausible explanation for routed's advantage here, not a proved auto failure
path. Conversely, routed's baseline-first design has no useful checkpoint on the
large regression cases. This is why neither ordinary runtime nor backup capability
alone proves that a strategy is safer on every table.

## 7. Completeness, limits, and files

**Done:** every requested dataset and edge-case category was tested. Real data
did not require a fallback substitute. Largest shapes were tested for all tasks;
the other edge cases were assigned to the tasks shown, not every possible
task/edge-case combination. Profile checks confirm six distinct numeric-looking
labels,25 classes,5,000 category levels, and recognized constant/duplicate/all-
missing columns. The high-cardinality case also has1,000 unseen test categories.

**Verified:** all145 valid outputs' requested metrics were recomputed from saved
labels and prediction files; classification/RMSE/MAE/R2/ARI also use independent
integer/standard-library arithmetic. AMI was rechecked with sklearn. Every input
hash and the49 input/reference separations were checked. All prior project files,
including the uncommitted third batch, match their before-check hashes.

**Not established:** official input/profile/score compatibility (rules unavailable),
Linux/assessment hardware timing (this is Windows/Python3.12.14), performance on
all larger shapes or combinations, statistical significance from three splits,
and the intended candidate after a worker was killed. No recommendation is an
official hackathon score estimate. No models were tuned using these test labels.

Complete dataset+seed tables: [classification](classification-table.md),
[regression](regression-table.md), [clustering](clustering-table.md).
Their CSV files have separate strategy×metric columns for spreadsheets.
[REPORT.md](REPORT.md) lists every individual loss, every worst dataset mean and
worst single seed. [stress-table.md](stress-table.md) contains all39 calls.
[verification.json](verification.json) records the audit. [README.md](README.md)
explains replay and report generation through the single `check.py` script.

The original baseline remains the default. No production change, commit or push
was made. The evidence supports conditional task choices and exposes substantial
large-table reliability gaps; it does not support a universal strategy switch.
