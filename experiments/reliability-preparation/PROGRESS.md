# Reliability and submission preparation

Checkpoint: `4c76b2fd83ad2db7f6c427f839858a2d135a05c2`.
Deadline: 2026-10-21 23:00 Asia/Beirut (20:00 UTC).
Freeze cutoff: 2026-10-21 15:00 Asia/Beirut (12:00 UTC).
Started: 2026-10-20 23:25 Asia/Beirut. Approximately 23h35 until submission.

Phase 0 complete: approved 45-file checkpoint exists. Local excluded evidence is
untracked and preserved. No push authorized. Every later commit needs approval.

Phase 1 completed and STOP CONDITION REACHED at approximately 2026-10-20 23:35
Asia/Beirut, after about 10 minutes. All nine specified combinations were traced
serially. Wide auto classification was interrupted in histogram binning, not
forest fitting. It had finished the validation forest and linear fits but no
final model. Await explicit user instructions before proceeding to repairs.

No production changes. No later phases run. No additional commits or pushes.
Detailed results: PHASE_1_REPORT.md, diagnosis-summary.json, phase-1-status.json.
Original source/config hashes and all 211 historical protected files match.
New verification/profile-reference scripts are preparation only, unexecuted and
not validated; do not mistake their presence for completed profile verification.

Stop if the principal cause is not forest fitting, both fixed forest and existing
linear recovery fail, or any normal-size existing-strategy predictions change.
No code/config/README/dependency changes after recorded freeze. Final failure
requires reporting and awaiting instructions without patching.

Announcement is the complete supplied specification. Metrics, concrete sizes,
machine-readable interface and hardware details remain unspecified.

## Authorized continuation

User explicitly extended scope to histogram guarding and equivalent label-membership
validation, alongside bounded forests and existing linear recovery. The earlier
non-forest stop was resolved by this authorization; historical diagnosis is retained.
Remaining stop conditions: both forest and linear fail, ordinary-size existing
strategy predictions change, or frozen final verification fails.

Phase 2: minimal changes implemented; 62 focused tests pass, including strict
membership parity, bounded sampling, mocked timeout/error linear recovery and
existing spawned checkpoint/deadline tests. Earlier test setup attempts failed
because a writable pytest temp parent was missing; corrected without changing tests.
The nine repaired CLI cases are now running serially, with separate output paths.

Phase 3: independent full-CSV profile reconstruction checked all 13 stress inputs
against all 37 surviving historical profiles. Every field passed; exact discrete
comparisons, numeric rtol=1e-10 / atol=1e-12. No production profiling function was
used as reference. Sample fitting cannot affect these full-data profiles.

Phase 2 initial repair check complete: nine/nine valid, full profiles checked,
zero emergency outputs, maximum 15.172s. All guarded forest attempts completed;
no natural linear retry was necessary. Histogram-guarded wide auto classification
selected existing LogisticRegression. Regression R2 .76588 (100k x50) and .31626
(5k x500), replacing mean-only recovery. Bounded classification accuracy .9075;
wide auto .931. Quality tradeoffs versus successful comparators remain to report.
Result validation failures also enter the single permitted linear retry; focused
mock coverage includes invalid payloads as well as error/timeout.

Phase 4 implemented: explicit submission mapping and default remains baseline.
README/model card distinguish supplied announcement from interface/resource/metric
assumptions and preserve classification objective conflict. No visibility change.

Next: full correctness suite, all 39 stress, all 108 real CLI equivalence checks,
submission mapping checks, source freeze, then final 57 submission invocations.

Full correctness suite passed: 295 tests, three expected warnings, 142.91s;
full-tests.xml and full-tests.txt retain evidence. The three-phase CLI campaign
has started serially and stops on any invalidity/equivalence failure. A genuinely
isolated clean-env (no system site packages) installed all requirements.lock pins;
clean-install-003.txt records successful installation. Its final smoke remains pending.

Phase 5 stress CLI verification completed successfully at 2026-10-20T21:05:12.854087+00:00. Per-run evidence: stress.jsonl.

## Paused at user request

- Completed: 295 tests; independent profile audit of all 13 stress inputs; nine/nine repairs and 39/39 stress runs valid with no emergency outputs; 86/108 real runs recorded. All recorded real runs passed historical prediction/model equivalence.
- Remaining: finish real-data equivalence, 49 submission mapping checks, record freeze, 57 final submission checks and isolated-environment smoke. No freeze exists; no new commit or push.
- Resume: read PAUSED.json; recheck current time against Oct 21 15:00 Lebanon freeze cutoff, inspect and preserve partial output before resuming campaign.py, then continue the approved sequence. Do not restart completed runs or overwrite interrupted evidence.

## Resumed 2026-10-21T05:24:10.039653+00:00

Verified checkpoint, expected file changes, all 134 completed verification prediction hashes, and available time. Interrupted evidence archived intact at real/real-clustering-iris-314/routed-0-interrupted-20261021T0523Z. No freeze exists. Resume remaining 22 real cases then submission delegation; do not repeat completed CLI calls.

Phase 5 stress CLI verification completed successfully at 2026-10-21T05:24:16.982603+00:00. Per-run evidence: stress.jsonl.

Phase 5 real CLI verification completed successfully at 2026-10-21T05:26:40.951159+00:00. Per-run evidence: real.jsonl.

Phase 5 submission CLI verification completed successfully at 2026-10-21T05:32:16.085042+00:00. Per-run evidence: submission.jsonl.

Phase 6 source freeze recorded at 2026-10-21T05:33:11.688889+00:00; source manifest 355fe705d2e1b69625004c7fdbd15571a6564a27f1a9dfc39b68a51832deeab1. Do not edit frozen files. Final submission verification and isolated-environment smoke remain pending.

Phase 6 frozen CLI verification complete at 2026-10-21T05:41:26.431603+00:00: 57/57 valid, no emergency outputs, maximum 28.375s. All frozen source/archive hashes intact. Isolated-environment smoke is the only remaining execution check. No code/config/README/dependency changes since freeze.

## Completed 2026-10-21T05:44:18.478603+00:00

All 57 frozen CLI calls and all three isolated pinned-environment smoke checks passed. Smoke predictions match exact frozen hashes. No post-freeze source/config/dependency/README/default changes. Original 145 valid historical prediction files and 205 other historical protected files verified unchanged. No commit or push after initial checkpoint. See COMPLETE.json, REPORT.md and CHANGED_FILES.json. Await approval for any further commit/push or new work.
