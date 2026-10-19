# Parallel work plan: submission readiness

> Priority update: the user requested a concrete grade-focused evaluation agent.
> That agent is now preparing independent analysis and actionable experiments in
> `reports/grade-evaluation/`. The readiness work below is a secondary backlog,
> not the current execution priority. Model implementation and timed experiments
> remain with the other chat.

Prepared 2026-10-19 from the active chat "Build dataset analysis program",
PHASES.md, reports/RESULTS.md, reports/VALIDATION_PLAN.md, README.md, and the
current provisional adapter. This is a proposed workstream; the checklist below
has not been executed. The other chat can continue without waiting for it.

## Current progress and division of responsibility

The other chat has completed the MVP, correctness reviews, expanded evaluation,
and initial model experiments. Its recorded integrated result is 63 passing
tests, 6 passing CLI smoke runs, and 48 valid broad-comparison CLI runs with no
fallbacks. Those figures describe the recorded source versions, not a fresh
verification of the clustering change currently in progress.

Its active agent is investigating a clustering regression: development noisy
blobs fell from baseline ARI 1.0000 to experimental ARI 0.6337. The latest update
describes testing K-means-based mixture initialization, keeping the same number
of mixture fits, followed by fresh-seed confirmation after freezing the change.
Baseline remains the documented default. Leave that implementation, validation,
and promotion decision with the other chat.

Phase 5 still needs the official interface/checker, nine practice datasets,
published size/environment limits, clean installation, and submission details.
Preparing these requirements is useful regardless of the clustering outcome.

## Proposed work: 35–50 minutes of documentation and static inspection

| Order | Work | New deliverable | Completion criterion |
|---|---|---|---|
| 1, 10–15 min | Map contest requirements to the existing adapter, CLI, and outputs | CONTRACT_GAP_MATRIX.md | Each requirement has local evidence, an explicit provisional assumption, and a verification action when official files arrive |
| 2, 10 min | Prepare installation and reproducibility checks | REPRODUCTION_RUNBOOK.md | Commands specify Python/environment, unique output paths, evidence to record, and criteria for a clean-environment pass |
| 3, 10–15 min | Prepare the release checklist | SUBMISSION_CHECKLIST.md | Covers team names, public URL, pinned dependencies, official checker, all nine practice tables, runtime/memory limits, and frozen submission commit |
| 4, 5–10 min | Review consistency and report prioritized gaps | A short handoff section in SUBMISSION_CHECKLIST.md | Clearly distinguishes verified evidence, pending checks, and missing official inputs; flags proposed changes without applying them |

All new deliverables belong in reports/parallel-readiness/. These filenames are
proposed outputs, not files already created by this plan.

### Contract gap matrix contents

- Input layout and metadata field names, feature types, target/ID handling,
  supervised prediction rows, and clustering row assignment.
- Output filenames, prediction encoding/order, cluster-count semantics, profile
  schema and missingness/duplicate definitions, exit codes, and error behavior.
- Official scoring objectives, timeout accounting, memory/CPU limits, and allowed
  dependency/Python versions. Mark unavailable details as unknown.
- Trace each local assumption to the current adapter, runner, profiler, or README.
  Prepare adaptation locations without editing them.

### Reproduction runbook contents

- Distinguish the verified local .venv312 environment, which reuses bundled
  packages, from a genuinely clean dependency installation, which remains pending.
- Inspect the existing workflow and lockfile statically; do not change either.
- Plan a separate fresh environment or disposable checkout after active timed
  experiments finish. Do not reuse or modify .venv or .venv312.
- Specify separate report filenames; never overwrite experiments/baseline.json
  or an existing comparison. Record source identity, package versions, commands,
  exit status, output validity, and wall time.
- Leave maximum-size and memory benchmarks pending until official limits exist
  and the other chat's timing runs have finished.

### Release checklist contents

- Team names and final repository URL supplied and checked before submission.
- Installation, entry point, assumptions, and reproduction instructions agree.
- Official checker and nine practice datasets pass under the official contract.
- Runtime and memory are checked on published maximum cases/environment.
- A final model strategy is chosen from the other chat's reviewed evidence and
  official metrics; this workstream does not select or promote a strategy.
- Public repository contents and the exact submitted commit are recorded.
- Publishing, committing, changing CI, and repository creation are separate
  actions; this plan only prepares the checklist.

## Boundaries that keep the work independent

1. Write only new documents under reports/parallel-readiness/. Do not edit
   src/, scripts/, tests/, main.py, README.md, PHASES.md, requirements.lock,
   pyproject.toml, .github/, existing reports, or experiment artifacts.
2. Use lightweight, read-only inspection. Do not run ML fits, pytest, benchmark
   subprocesses, installations, downloads, or background jobs during the other
   chat's measurements; shared CPU, memory, and disk activity can distort timing.
3. Do not change environments, process settings, dependencies, branches, or Git
   state. Do not interrupt or message the other chat as part of this plan.
4. The evaluator currently fingerprints main.py, src/**/*.py, and scripts/*.py.
   This documentation directory lies outside those paths, so these planned
   additions do not change that fingerprint.
5. Treat reports as versioned evidence. Recheck progress before any later code
   changes or expensive validation; do not infer that a running experiment has
   completed from file presence alone.

## Handoff and later work

The immediate result is three reviewable readiness documents with prioritized
gaps. No dependency on the clustering experiment is required to prepare them.
When the model work finishes, combine its final evidence with this checklist.
Execute clean installation and runtime checks afterward, in isolation. Adapt
the contract and scoring only once the official specification is available.

This plan does not establish official compatibility, clean-install success,
hidden-test quality, or worst-case timing compliance.
