# Stage 1: clustering representation study — approval required

Prepared 2026-10-22. Audit and proposal only; no fitting, downloads, or model changes.

## Protected starting state

- Branch: `codex/clustering-representation-study`, created from verified backup `36e0bf421eb6dc8615e781aa0709a8017c073777`.
- `main` remains at that commit. There were no tracked/staged changes before branching. Existing untracked datasets, environments, references, predictions and other evidence were preserved.
- All 65 source/configuration/test/document hashes in the existing reliability freeze still match. Its old Wednesday deadline/cutoff is historical, not the deadline for this study.
- No commit, push, merge, cleanup, visibility change, default change or model execution is authorized by this stage.

## Supplied files: verified facts and roles

Exact SHA-256 hashes and sizes are in `supplied-files-audit.json`. Inspection read the originals without modifying them.

| File | Verified contents | Proposed role and handling |
|---|---|---|
| ai4i2020.csv | 10,000 rows, 14 columns; no parsed missing/blank values; Machine failure counts 9,661 / 339 | Later classification only. Before that evaluation, exclude UDI, Product ID and TWF/HDF/PWF/OSF/RNF; preserve identifiers separately. The five failure flags are outcome proxies: their Boolean OR matches the target in 9,973 rows, not all rows. Product ID's first letter duplicates Type in every row. Retain Type and the five physical measurements. Failure is not assumed to define natural clusters. |
| Mall_Customers.csv | 200 rows, 5 columns, no missing/blank values | Include as an unlabeled real mixed-feature clustering example from the announcement. Exclude CustomerID; retain Gender, Age, Annual Income (k$), Spending Score (1-100). Numeric scaling and categorical encoding stay as implemented. Proposed fixed k=5 is an evaluation assumption requiring approval, not a supplied requirement or discovered truth. |
| Mall_Customers (1).csv | Byte-identical to Mall_Customers.csv | Do not use or count separately. Preserve original. |
| Telco CSV | 7,043 rows, 21 columns; Churn No=5,174, Yes=1,869 | Later classification only. Keep customerID outside features. TotalCharges is text with 11 whitespace-only entries; numeric conversion makes exactly those 11 missing, with no additional invalid nonblank values. Propose numeric tenure/MonthlyCharges/TotalCharges and categorical service/demographic indicators, including SeniorCitizen. Keep 'No internet service'/'No phone service' as real categories. No classification change here. |
| data_description.txt | 523 lines of housing-feature documentation, no observations | Reference only. Existing real-data evaluations already include OpenML house_prices 42165: 1,460 rows, 80 features, 43 categorical. Of 76 field headings extracted from this description, 74 match that input schema. This does not add a dataset or a new regression family. |

For Mall, do not report reference-label ARI/AMI. Report valid k occupancy, sizes/repairs, full-process time, recovery, silhouette/Calinski-Harabasz/Davies-Bouldin and partition stability. Evaluate internal metrics in a fixed incumbent representation as well as each fitted representation, clearly labeled: changing the metric's distance space can itself change its score. Use five fixed 80% subsamples with fixed seeds, comparing co-assignment stability on shared rows. Such stability is descriptive, not correctness. No k or feature-subset search. The clean input is a no-missingness control; a random-30% numeric-mask variant is a condition of Mall, not a second family.

## What the existing evidence actually establishes

- Initial and broad synthetic studies covered spherical/anisotropic/noisy blobs, unequal sizes/variances, correlated ellipses, moons/circles, outliers, irrelevant features and mixed categories. These are inspected development evidence now, even where old reports call seeds 'confirmation'.
- First batch: six mechanisms, three seeds, five mask conditions, four policies (360 calls). Removing indicators had 61 wins, eight losses, 21 ties; worst ARI loss was .32086. Its feature-dependent masks differ from group-label-dependent masks.
- Second batch: exact eight-loss replay, geometry versus K-means, initialization checks and random versus group-informative masks. Removing indicators can lose .45235 ARI. Converged GMM/BIC can prefer a partition that disagrees with the reference groups. Representation losses also occur with K-means, so GMM is not the sole explanation.
- Third batch already compared amplitude 1 and .25 on 24 conditions (four mechanisms, two seeds, three masks). Quarter amplitude gained .07015 ARI/.06745 AMI overall against current geometry, but informative-mask AMI fell .01919; worst ARI/AMI changes were -.21520/-.26502. The frozen acceptance rule failed. Do not repeat this and call it new confirmation.
- Real clustering: iris/wine/digits, three seeds, baseline/auto/routed, all inspected. Mean ARI .68233 to .77803 and AMI .73108 to .81293 for auto; all gain came from iris. Their class labels are external proxies, not proven natural group truth. Auto and routed matched in these saved cases.
- The repaired runtime evidence includes 100,000 x 50 and 5,000 x 500 clustering inputs and a constant/duplicate clustering input. These test engineering validity, not broad clustering recovery or large masked representation behavior.

Sources: `reports/clustering_experiments.md`, `reports/clustering_improvement.md`, `reports/first-batch/clustering.md`, `reports/second-batch/clustering.md`, `reports/third-batch/clustering.md`, `reports/grade-evaluation/LATEST_EVALUATION.md`, `scripts/datasets.py`, `scripts/batch_clustering.py`, `scripts/second_clustering.py`, `scripts/third_clustering.py`, `experiments/real-data-check/{manifest.json,clustering-table.md,check.py}`, and the reliability reports/freeze.

## Coverage matrix

Covered means recorded evidence exists, not that the method performs well. Conditions complement one another; no dataset must cover every row.

| Property | Existing inspected evidence | Gap / proposed action |
|---|---|---|
| Equal group sizes | Spherical/correlated/curved generators; iris | Retain controls; no new equal-blob family needed. |
| Unequal sizes | [100,210,350] and [40,80,120] unequal-density generators | Retain; not evidence about extremely rare groups. |
| Small/rare groups | Rare groups mentioned as sampling risk; minimum prior unequal group about 15–17% in these generators | Add fixed 90/9/1% synthetic condition; report per-reference-group recall as well as ARI/AMI. |
| Clear and overlapping groups | Separated blobs, noisy blobs, overlapping ellipses/moons | Retain one clear and one overlapping control. |
| Different shapes/densities | Anisotropic, ellipses, unequal variance, moons/circles | Already covered; no new estimator or geometry sweep. |
| Numeric and mixed categorical/numeric | Synthetic mixed_clusters/mixed; real numeric iris/wine/digits | Add all-feature Mall, with unlabeled limitations. Fresh mixed real confirmation remains a gap. |
| Different scales / irrelevant features | Mixed numeric axis x1000, noisy_blobs, missing_noise axes | Retain representative cases. |
| No missingness | Real numeric datasets and synthetic clean controls | Retain; add clean Mall. |
| Random missingness | 10/30% masks; intrinsic 50/70% masked irrelevant axes | Already broad development coverage; add matched fixed masks only, not a new family claim. |
| Informative missingness | Feature-dependent masks and label-dependent 65%/3% masks | Retain both mechanisms; prior quarter-amplitude losses are mandatory sentinels. |
| Groups primarily/entirely defined by masks | No explicit missingness-only generator found | Add mask-only reference groups with identical observed values; include absolute recovery criteria. |
| Structured masks unrelated to intended groups | Independent random missing noise exists, but no explicit independent site/batch block-mask mechanism found | Add nuisance site/batch block masks crossed with known value-defined groups. |
| Constant/duplicate columns | Unit tests and constant/duplicate stress, including all-missing columns elsewhere | Add observed-constant/different-mask tests; exact duplicates with same mask; same values with different masks; categorical missingness. |
| Larger/wider runtime | 100k x 50 and 5k x 500 clean clustering CLI cases | Retain; add one frozen 30% mask to each for indicator-width cost. Not official size limits. |

Source-level information-preservation hypothesis: `preprocessing.py:53` uses `nunique(dropna=False)`, so a column with one observed value plus missing rows is not removed as constant. Exact duplicate removal compares full cleaned series, including missing positions (`:62-66`). Numeric imputation adds indicators (`:91`), and amplitude scaling happens after standardization (`clustering.py:17-39`). Thus those paths do not establish an existing information-loss defect. Focused tests must trace the actual transformed columns before any correction. All-missing and fully constant columns have no varying mask. Categorical missingness is encoded as a category; the numeric amplitude control does not reweight categorical missing tokens. Rare-category grouping is an additional audit point, not permission for unrelated tuning.

## Inventory and acquisition plan

### Development — all already inspected results remain development

- Original synthetic mechanisms, all old 'confirmation' seeds including 8081/2027/4099, all three development batches, iris/wine/digits and existing stress fixtures.
- Retain the exact eight historical losses plus the third-batch 24-case inputs as sentinels. Reuse saved input bytes, not regenerated cross-platform approximations. Existing scores are context; new baseline measurements use the verified checkpoint.
- Mall (one copy) is new to the recorded model evaluations, but its schema is now inspected. Assign it to descriptive development, never untouched recovery confirmation.
- Proposed new development generators, 600 rows, k=3, seeds 104729/104743/104759: (a) mask-only signal, 12 numeric columns with observed value 7 and group-specific four-column mask blocks, 5% independent mask-bit flips; (b) value-defined Gaussian groups plus eight nuisance numeric axes with site-dependent 80%/5% block masks, site independent of intended group. For mask-only, a bit flip changes missing to observed or vice versa; all observed values remain 7. For nuisance, fixed centers [-4,0]/[0,4]/[4,0], unit numeric noise, random independent three-site assignment; independent normal nuisance values. Each mechanism has equal-size and 90/9/1% variants. These are two mechanisms with multiple conditions, not twelve independent families.
- Label-dependent masks exist only in explicitly disclosed generators. References are saved outside input folders; no references go to fitting, BIC, resource admission or production selection.
- Retain the existing clean large fixtures; propose one fixed random-30% mask per shape using seed 104729. These are runtime conditions, not new natural families.

### Confirmation — not yet enough untouched material

No existing scored result is eligible for untouched confirmation. A name/provenance search of tracked scripts, tests, reports and text/JSON experiment records found no prior Palmer penguins, Glass Identification or UCI Seeds use. This is repository-bounded evidence, not proof of every previous human interaction. Existing untracked real-data manifest inventory was also inspected; its families are listed above and include housing, not those proposed sources.

After approval only, acquire two proposed independent real families: **Palmer penguins** from the publisher's maintained data repository (mixed morphology/sex/island, with species held outside inputs), and **UCI Glass Identification** from the UCI repository (numeric measurements, class held outside inputs and any identifier excluded). Verify source metadata, license, dimensions, missing tokens, hashes and any alias/prior-use overlap before registration; these acquisition details are pending, not verified claims. Species/glass class are defensible external proxies with semantic limitations, not declarations of unique natural clustering truth. k is the number of distinct supplied reference classes, fixed before fitting. No outcomes may determine exclusions or k. No data have been downloaded.

Acquire/register both before development results, keep their scored results unopened, and freeze exact feature lists/types before the first model run. Penguin features proposed: bill length/depth, flipper length, body mass, sex and island; exclude species, row IDs and year. Glass: all documented non-ID measurements. Confirm actual schema before registering; stop for approval if incompatible rather than silently substituting a favorable dataset. Freeze three model seeds 130003/130021/130027; clean, random-30%, informative-65/3%, and independently generated nuisance-mask conditions are conditions of each original family. Group-dependent corruption uses evaluator-only references explicitly; it never reaches production code. Results from a masked familiar family are not independent real families. No repeated dataset/seed/condition is counted twice.

Two independent families still provide limited generalization evidence; neither guarantees natural missingness coverage. New seed-only mask-only/nuisance cases can check repeatability but do not fill that independent-family gap. If acquisition, freshness or time is insufficient, report insufficient evidence for promotion and retain the incumbent; do not replace missing confirmation with familiar iris/wine/digits. A supplemental Seeds dataset is a possible later addition only with separate approval, not an outcome-driven substitute.

## Registered experiment after approval

1. Save immutable input bytes/hashes, separate references, feature schemas, generators/masks, seeds, k, settings, environment/dependencies/source hashes and run order before fitting. Add no raw arrays to a proposed Git checkpoint. Keep everything under this study's new local output paths without replacing historical files.
2. Run unchanged checkpoint first on development inputs. Compare full-strength 1, quarter-strength .25, and no numeric indicators, with identical input/seed/k/estimator settings and selection budgets. Existing optional engine controls already implement these representations. Reuse them rather than rebuilding preprocessing. The omitted policy is `clustering_missing_indicators=False`, not amplitude 0 (which the current validator rejects).
3. Primary comparisons use the same existing geometry engine and unchanged eligibility/BIC/covariance/restart settings, with current baseline K-means results separately labeled. Do not compare BIC across different representation spaces. Save selected estimator, per-group sizes, convergence and repairs. Use matched fixed run order balanced across policies, serial CPU execution, one thread where current runner enforces it. Direct engine timings do not count as full CLI timing.
4. Audit transformed information with tiny deterministic cases before any correction. If none is lost, make no preservation fix. If a loss is demonstrated, test a clustering-only opt-in correction independently: incumbent, correction only, amplitude only, combination (include omitted representation with/without correction where applicable). No global cleaner change or silent cross-task effect. Any correction expands the run count; time-box rather than reduce validation covertly.
5. An isolated study worker can call the existing configurable engine for controlled representation measurements without changing production dispatch. Use the existing supported opt-in routed config for end-to-end checks, preserving checkpoint recovery; explicitly record routed checkpoint-versus-final origin. Do not infer CLI results from direct calls. If existing plumbing cannot expose an exact comparison, propose minimal opt-in plumbing before editing it; do not change baseline/auto/submission semantics.
6. Select at most one candidate using only registered development evidence and the gates below. Prefer no change if both alternatives fail. Freeze its exact source/config and policy before opening confirmation scores. Record every attempted fit, timeout and failed policy; no parameter sweep or unreported rerun. A second revision makes those confirmation results development and needs new approved confirmation families. Mixed wins remain mixed; no adaptive selector.

## Fixed proposed gates (diagnostic, not official scoring)

- Nonnegotiable: all outputs valid, complete full-data profiles, no new emergency recovery, no unexplained empty-group repair increase. Report every repair and failure. All runs under 60 seconds full-process; use existing 50-second configured budget/reserves. Target <=40 seconds; investigate >45 seconds. Candidate paired median overhead must be <=max(1 second, 10% of incumbent median), with no deadline/recovery regression. Report available peak-process memory evidence and limitations; do not invent machine-wide memory observations.
- Average seeds within each original family, then average families equally; report missingness strata separately. Do not mix Mall internal measures into ARI/AMI means. Report every paired result and worst losses, not just averages.
- Eligibility to recommend independent confirmation: family-balanced ARI gain >=.02, AMI gain >0, nonnegative mean ARI and AMI changes separately on random and informative masks; no original-family mean ARI/AMI loss >.02. Flag every individual ARI loss >.10; any unexplained such loss blocks a general promotion recommendation. Rare-group matched recall must not fall >.05 on average within its condition. Match groups to references offline only; report unmatched/empty groups.
- Missingness-only: on balanced, clearly encoded fixed-mask conditions require absolute mean ARI and AMI >=.80, not merely a tie with a failing incumbent. On the rare-group variant report absolute ARI/AMI and smallest-group recall separately; average matched recall >=.80 and smallest-group recall >=.70 are proposed guards against ignoring the rare group. Failure is evidence against a broad claim, even if aggregate scores improve.
- Confirmation must pass the same quality/runtime/validity gates, evaluated at the family level and per missingness stratum, without another revision. Small samples require explicit uncertainty and coverage limitations; passing is grounds to consider promotion, never automatic approval. If no candidate passes development, do not open confirmation scores merely to find a winner.
- Mall: valid k=5, finite internal metrics when defined, unchanged IDs excluded, five fixed subsampling checks; no external-correctness acceptance claim. No missingness clean control should change with amplitude. Internal stability/quality is reported, never used to override failed reference-label gates.
- Isolation: classification/regression prediction hashes and selected models must match saved checkpoint comparators; disabled experimental controls must match clustering within each existing baseline/auto/routed/submission strategy. Profiles must match full-data independent checks (exact discrete fields; existing documented numeric tolerances). Reference-label perturbation must not change fitted preprocessing, selection, resource admission or predictions. Preserve worker isolation, deadlines, checkpoint transfer and emergency recovery. Unexpected changes trigger a stop and report, not unrelated fixes.

## Proposed schedule and pause

At audit time: Thursday Oct 22, 2026, approximately 22:44 Lebanon time. Saturday means Oct 24; exact deadline time remains unconfirmed. No Wednesday cutoff is reused. Reserve the final eight hours relative to the confirmed Saturday deadline for frozen verification, submission checks and user-approved publication; current private visibility stays unchanged.

| After approval | Work / saved progress note |
|---|---|
| 0:00–0:30 | Provenance/acquisition checks, immutable registration and unchanged baseline. If acquisition prevents completion, report reduced scope before running a partial campaign. |
| 0:30–1:10 | Focused information-preservation tests; only a proven minimal opt-in correction. |
| 1:10–2:10 | Matched three-representation development measurements, including Mall descriptive checks. |
| 2:10–2:30 | Family/stratum report, failed-policy ledger, choose at most one candidate and freeze. |
| 2:30–3:10 | Open reserved confirmation only if eligible; otherwise preserve it. |
| 3:10–4:00 | Correctness/default equivalence, CLI deadline/recovery probes, final report and exact changed-file inventory. |

This is an approximately four-hour cap, not permission to omit checks. Expensive cases run serially. If a correction, acquisition or runtime checks make this infeasible, stop and report the remaining work; do not silently broaden the campaign or consume the final verification buffer. Save a short progress note after each stage. Any separately approved integration requires cross-task/runtime verification again. No commit/push/merge/default change is included.

Approval requested for k=5/all-four-feature Mall handling, the two new synthetic mechanisms and rare conditions, the proposed acquisition/confirmation inventory, and these gates/schedule. Confirm the exact Saturday deadline in Lebanon time before scheduling the protected final buffer. STOP here pending approval.
