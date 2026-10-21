| Phase | Timing and result |
|---|---|
| Checkpoint | Approved 45-file checkpoint retained; no subsequent commits |
| Diagnosis | Nine serial calls; forest, histogram binning and validation costs isolated |
| Repair | Nine/nine valid; 8.92–15.17s; no emergency outputs |
| Profiling | All fields independently checked on all 13 stress inputs |
| Pre-freeze validation | 295 tests; 39 stress +108 exact real-data comparisons +49 submission checks passed |
| Freeze | October 21, 8:33 a.m. Lebanon; more than 14 hours before submission |
| Final verification | 57/57 passed; maximum 28.38s; three clean-environment smoke checks passed |

The bounded forest uses exactly 16 trees, at most 20,000 training rows, sqrt feature selection,
deterministic sampling and class preservation. A failed attempt can use the existing linear model
within the same absolute deadline; emergency mean/majority remains last. All eight initially guarded
forests completed, so no natural linear retry was required. Forced error/timeout/invalid-result
recovery checks passed. The wide auto classifier now skips expensive histogram binning and completes
with its existing linear candidate. Distinct-label validation preserves accepted/rejected values.
No model-family, weighting, threshold or clustering tuning was added.

| Task / fixture / strategy | Original CLI | Repaired CLI | Result |
|---|---:|---:|---|
| classification 100000x50 baseline | 50.90s | 9.34s | Valid learned predictions |
| classification 100000x50 routed | 54.01s | 12.30s | Valid learned predictions |
| regression 100000x50 baseline | 47.33s | 8.92s | Valid learned predictions |
| regression 100000x50 auto | 47.41s | 9.41s | Valid learned predictions |
| regression 100000x50 routed | 47.58s | 12.09s | Valid learned predictions |
| regression 5000x500 baseline | 47.37s | 11.66s | Valid learned predictions |
| regression 5000x500 auto | 47.38s | 11.84s | Valid learned predictions |
| regression 5000x500 routed | 47.52s | 15.17s | Valid learned predictions |
| classification 5000x500 auto | 47.56s | 12.17s | Valid learned predictions |

The detailed diagnosis stage timings, interrupted calls and unavailable measurements remain in
PHASE_1_REPORT.md and diagnosis-summary.json. Original timings above are uninstrumented saved runs;
diagnostic tracing has overhead. Output writing took .06–.25s, so its deadline error was not proof
that serialization consumed the budget.

**Usefulness and costs.** All 30 supervised stress runs beat their corresponding mean/majority dummy
on RMSE or accuracy. Large-regression R2 is .766 (100k x50) and .316 (5k x500), replacing mean-only
outputs. Wide auto classification accuracy is .931. The bounded large classifier loses accuracy:
.9075 versus .9565 for the successful 96-tree diagnostic run and .9595 for retained auto. No losses
occurred against previously successful original stress outputs within their existing strategies;
the separate diagnostic forest comparison is explicitly retained. No post-repair run exceeded 45s
or needed emergency recovery. Maximum across pre-freeze and final verification: 28.61s.

**Preservation.** All 108 original real-data prediction hashes and selected models match. All 36
frozen real submission cases match the appropriate historical strategy; all 21 final stress/repeat
calls match their strategy counterparts. The isolated pinned environment also reproduces three
exact prediction hashes. Every profile field uses full training data, never fitting samples;
discrete fields match exactly, numeric values use rtol=1e-10 and atol=1e-12. Historical data and
reports remain local and unchanged. These are local diagnostic results, not official hackathon points
or certification of unspecified assessment sizes/hardware.

**Files.** Exact runtime/document/test and nine reusable-script lists: CHANGED_FILES.json. Main edits:
README.md, MODEL_CARD.md, src/unknown_dataset/checkpoints.py, models.py, runner.py, supervised.py;
new src/unknown_dataset/reliability.py and tests/test_reliability_preparation.py. Configs, shared
preprocessing, production profiling, evaluator framework and historical reports are unchanged.
All new verification outputs are under this separate folder. Large data, caches, raw outputs and
the isolated environment are local evidence, not proposed repository contents.

**Frozen state.** Initial checkpoint: `4c76b2fd83ad2db7f6c427f839858a2d135a05c2`.
Source/configuration manifest SHA-256: `355fe705d2e1b69625004c7fdbd15571a6564a27f1a9dfc39b68a51832deeab1`.
Archive SHA-256: `ca5a8964cb30aa41ac6d9df037d4c1cabf10a590c6b3f34c307d394c3022b830`.
Individual hashes, 17 pinned versions, Python/OS and input hashes: freeze.json. No frozen file changed.

```sh
python main.py --input INPUT --output OUTPUT --strategy submission --time-budget 50 --seed SEED
```

Submission remains explicitly selectable and non-default: classification baseline; regression and
clustering auto. CLI/API defaults remain baseline. The announcement is the supplied specification;
metrics/normalization/failed-run penalties, interface, concrete size limits and hardware are unspecified.
Our schema, command, metrics, mapping and resource limits are documented interpretations. Classification
baseline remains a conservative choice with a measured balanced-accuracy disadvantage. The repository
remains private and the third member name is unconfirmed; publication/push needs separate approval.

The only authorized checkpoint made remains `4c76b2fd83ad2db7f6c427f839858a2d135a05c2`; no later commit, push, visibility change
or discarded evidence. Working changes are uncommitted and excluded evidence remains untracked.
Proposed next checkpoint, subject to approval: `checkpoint: repair large-input reliability and prepare explicit submission`.
