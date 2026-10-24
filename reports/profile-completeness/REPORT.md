# Profile completeness: additive reporting only

Branch `codex/profile-completeness`, based on unchanged main `aada866729cf6f6084d1d5f09dacbb45fae1cd33`. Clustering experiments remain on their own branches. No model, preparation, selection, prediction policy, default, dependency or execution-limit changes; README untouched.

## Existing fields and additions

Existing profiles already report per-feature missing counts/fractions and unique counts, numeric summaries, constants/duplicates, and target summaries/distributions. These are retained with their original meanings. schema_version remains provisional-1; the additions are our interpretation of useful reporting, not an organizer-prescribed schema.

- Each categorical feature gains `categorical_summary.top_values`: up to 20 nonmissing value/count records, descending frequency; ties use first occurrence in the full training column. Values retain lexical spelling. Counts use every training feature row, not a fitting sample. Unused categorical dtype levels are not observations.
- `categorical_summary.omitted_category_count`: number of observed nonmissing categories excluded from top_values.
- `categorical_summary.omitted_value_count`: number of nonmissing training rows belonging to those omitted categories.
- Missing counts remain explicitly in the existing per-column `missing_count`, without a duplicate field in the summary. Sum(reported counts)+omitted_value_count+missing_count=row_count. Existing unique_count covers all nonmissing categories.
- Clustering profiles gain root `requested_cluster_count`, passed from validated metadata.n_clusters. It is a request, not an observed/inferred group count. Observed groups remain model metadata in run.json. Supervised profiles omit the new root field. The public profile helper accepts optional keyword-only n_clusters.
- Categorical output is bounded by 20 records per feature, independent of cardinality; individual value-string length is not newly capped or truncated. Target distribution behavior is unchanged. Full training means the existing contract feature table; ID/target exclusions retain their existing behavior.

## Independent correctness and equivalence

- 63 distinct existing input folders checked with independently calculated full-data frequency Counters and the existing separate profile-reference implementation. No production profiling helper is used to derive expected new values. Discrete fields exact; existing numeric fields rtol1e-10/atol1e-12.
- Strip only the two explicitly added field kinds, then compare every existing profile field against independent reconstruction and the saved original profile. All match; the historical checker/evaluator files are unchanged.
- 120 distinct saved CLI configurations rerun serially, covering all 164 saved runs from classification-runtime-repair and independent-current-review-20261023. Only byte-identical argument configurations were deduplicated; default/explicit strategy arguments remain distinct. Every prediction SHA-256 and selected model/candidate/selection metric matches all corresponding saved originals.
- All 120 calls valid learned outputs, zero emergency fallback; maximum complete-process CLI runtime 44.638s (<45s, within unchanged 50s invocation budget). No new quality experiment, quality scoring, dataset generation or download.
- Categorical cardinality observed in saved fixtures reaches 5000; maximum published records per feature is 20. An independent 45,000-row unit check covers 5,001 categories and a most-frequent category occurring only after row30,000, plus 5,000 missing rows, demonstrating full-data counts and bounded output.
- Focused profiling/integration/reliability/default/leakage tests: 63 passed. Full suite: 323 passed. Includes lexical/missing distinctions, all-missing/empty columns, unused categorical levels, deterministic ties, requested count validation and actual CLI publication.

## Runtime evidence

One paired old-main/new profile-plus-JSON-serialization measurement per existing input, with the same loaded full table: median added time -0.000154s, largest added time 0.330813s. Predeclared review thresholds were median<=.1s and maximum<=.5s. No material attributable profiling overhead was observed; these are local single-pair measurements, not a cross-machine performance guarantee.
Complete CLI runtimes versus historical saved runs: median change +2.923s, largest increase +12.556s. Historical whole-process comparisons include model/runtime variability and are not isolated profile-cost measurements. Every checked prediction and selection remained unchanged.

Two existing probes with historical timing increases were additionally run on current main and the profile branch, serially with identical inputs, seeds, arguments, dependencies and 50s budgets; no held-out quality scoring or tuning. These four control calls are separate from the 120 equivalence calls:
| Existing probe | Current main seconds | Profile branch seconds | Difference |
|---|---:|---:|---:|
| probe-10000-100 | 43.209 | 41.938 | -1.270 |
| probe-45000-50 | 18.123 | 18.669 | +0.545 |

All four control calls retained saved prediction hashes/selected models and independently correct profiles, without emergency recovery. Single paired timings are local diagnostics, not statistical proof of identical execution time. The wide numeric probe has no categorical additions and produces the same profile contents; its historical slowdown cannot be treated as evidence of categorical-counting cost.

## Exact candidate files and preservation

`src/unknown_dataset/profiling.py`, `src/unknown_dataset/runner.py` (profile call only), `tests/test_profiling.py`, `tests/test_cli.py`, and this report. No README edit or evaluator framework modification.

Frozen source/config/test hashes are preserved in local candidate-manifest.json. Main and all tracked files in 5 pre-existing checkouts retain hashes; 25343 old local evidence size/mtime records are unchanged. Existing raw files, profiles, predictions and reports were read without overwriting. New output arrays/caches stay local and are excluded from proposed repository contents.
The local check.py, runtime_check.py and finalize.py harnesses, case inventory, cached audits, progress/PIDs, paired timings and final-verification.json live outside the candidate worktree. They permit resumption without repeating completed runs. Source/inputs were frozen and checked before calls. No commit, merge or push; staging is for review only.

Limitations: machine-readable profile fields and metadata are implementation choices, not verified official grading requirements. Added fields can require strict external schema consumers to accept additive keys; no claim of official compatibility. The 20-record bound does not bound individual category string lengths. Runtime evidence is from this machine and existing fixtures.
