# Classification runtime repair — review summary

Branch `codex/classification-runtime-repair`; main and branch HEAD remain `5b12dc2d54f0b0e2807ee823630b61a9ed879b5a`. The six integration edits were preserved as exact bytes under local `integration-before/` before creating this branch. No staging, commits, merges or pushes.

## Change and rationale

The new training-only admission guard activates for baseline/auto/submission classification with >20,000 rows, categorical features and a conservative prepared-width upper bound implying >2,000,000 full-training cells. It includes numeric missing indicators and capped one-hot categories. The rule was registered before validation in RESOURCE_RULE.md and never revised. Constants/duplicates can overestimate the bound; this is an engineering assumption, not an official size limit.

Admitted tables use the existing 16-tree/sqrt-feature/depth-24 forest on <=20,000 deterministic class-preserving rows, then existing linear recovery if needed. This avoids selecting a small-sample histogram whose full-data matrix cannot pass its cap, followed by an unrestricted 96-tree refit. Both CLI parent and public API use the same admission rule. Existing parent deadline/reserves, preprocessing, estimators, weights, metrics and caps are unchanged. The public API still requires its caller to enforce a hard process deadline. Numeric-only inputs and regression/clustering do not gain a new rule. The selector implementation itself was not changed.

## Exact failed fixture, paired execution results

| Strategy | First wall time | Repeat wall time | Output |
|---|---:|---:|---|
| auto | 12.908s | 13.646s | Learned bounded forest; identical predictions/profiles/selection |
| baseline | 12.905s | 15.501s | Learned bounded forest; identical predictions/profiles/selection |

| Metric | Earlier emergency / training-majority predictor | Repaired (both strategies) |
|---|---:|---:|
| accuracy | 50.85% | 77.70% |
| balanced_accuracy | 50.00% | 77.72% |
| macro_f1 | 33.71% | 77.70% |
| Recall: negative | 0.00% | 78.74% |
| Recall: positive | 100.00% | 76.70% |

Previously auto/baseline took 47.483/47.461s in the instrumented diagnosis, returned EmergencyMajorityClass and never completed their forests. New calls use the actual CLI without diagnostic monkeypatches. All four beat majority on accuracy, balanced accuracy and macro F1; no emergency or linear recovery was needed. Existing forced-failure tests verify learned linear recovery separately. Full-data profiles passed independent reconstruction (exact discrete fields; floats rtol=1e-10/atol=1e-12).

The 77.70% accuracy is a useful reliability improvement over emergency output, not proof that a bounded forest is the best feasible classifier. No completed unrestricted forest comparator exists for this failed fixture, so its potential quality loss is unknown. This batch did not tune or compare new model alternatives.

## Other frozen execution probes

| Shape | Auto run 1 | Auto run 2 | Accuracy / BA / macro F1 |
|---|---:|---:|---|
| 20000-20 | 13.323s | 13.371s | 0.9180 / 0.9177 / 0.9178 |
| 10000-100 | 30.622s | 30.985s | 0.9225 / 0.9225 / 0.9225 |
| 50000-30 | 20.202s | 19.491s | 0.9005 / 0.9004 / 0.9004 |

These are the unchanged frozen seed-47131 inputs, not newly generated or independent generalization evidence. Each repeat retained prediction/profile hashes and model selection. The 20k numeric case also matches the earlier integration prediction hash. Earlier test/run results were not substituted for checks affected by changed source hashes.

## Preservation and integration checks

- 20 classification +18 regression/clustering historical cases: exact saved auto prediction hashes, selected models and profile hashes retained under default submission. Any mismatch would have stopped the serial driver.
- Explicit baseline/auto/routed and submission checked against saved predictions on wine, diabetes and iris. Omitted strategy and explicit submission also match repaired auto on the 45k mixed fixture. Focused API tests match default/submission/auto across all tasks.
- 39/39 original stress strategy combinations: learned, valid, complete independently reconstructed profiles, no emergency. Maximum 20.698s. Stress differences versus the successful reliability checkpoint: 0 cases; detailed metrics/deltas in local/stress-quality-comparison.json. No cheaper model was accepted on runtime alone.
- 309 full-suite tests passed, zero failures/errors/skips. Focused set: 46 passed, covering resource boundaries, encoding width, class-preserving sampling and forced classification forest-to-linear recovery with the same absolute deadline.
- Across all 101 fresh verification CLI calls: valid/complete profiles, zero emergency outputs, maximum full-process runtime 30.985s. Independent scoring/reconstruction and fixture hashing are outside the timed CLI invocation.
- 39 original protected source/report hashes and 12,457 old local evidence size/timestamp records retained. Diagnostic evidence and original raw inputs were not overwritten. Main, configs, pinned dependencies, selector/weighting code, shared preprocessing/profiling, clustering and evaluator files remain unchanged.

## Limits and recommendation

This fixes the diagnosed resource gap with useful learned predictions on this machine. It is not official-score evidence or certification of unspecified assessment hardware/sizes. No fresh confirmation campaign was run. Conservative width estimates may admit some inexpensive redundant tables, and successful larger mixed inputs newly caught by the rule can change predictions. All measured normal-size and original stress comparisons are reported above; unseen-input equivalence cannot be guaranteed. The numeric 10k x100 probe has less headroom (~31s) than the repaired mixed fixture. Keep the candidate on this branch for review.

The earlier independent review also recorded a Bank baseline timeout. Its successful auto run used 33,908 rows and 50 encoded features (1,695,400 cells), below this new guard. That separate historical baseline failure is not claimed fixed; Bank/Dry Bean were not rerun in this batch. No dataset-specific exception or additional threshold trial was introduced.

The inherited approved integration mapping remains submission -> existing auto for all tasks, with balanced-accuracy classification selection outside fixed resource admission. The macro-F1 challenger is not included. Improved class recognition can cost accuracy/precision, and the official grading metric remains unknown.

## Exact proposed commit files

- `MODEL_CARD.md`
- `README.md`
- `experiments/classification-runtime-repair/REPORT.md`
- `experiments/classification-runtime-repair/RESOURCE_RULE.md`
- `src/unknown_dataset/models.py`
- `src/unknown_dataset/reliability.py`
- `src/unknown_dataset/runner.py`
- `tests/test_classification_runtime_repair.py`
- `tests/test_reliability_preparation.py`
- `tests/test_submission_auto_integration.py`

These include the preserved integration source/docs/tests plus this repair and its concise rule/report. Local check.py, continue_checks.py, summarize.py, resume notes, preservation copies, raw inputs, reference/prediction arrays, caches and bulky outputs are excluded from this proposed commit. Await explicit approval before staging or committing; no push is authorized.
