# Revised-plan implementation run

User authorized continuing implementation after the plan-only revision. The baseline default
and provisional-metric gate remain unchanged. Missing official rules are not replaced by assumptions.

## Ownership
- Policies agent: routing/policies/config; routed supervised execution; experimental categorical counts;
  explicit metrics, configurable near-tie handling, and focused tests.
- Runtime agent: worker/checkpoints; real process failure/timeout recovery; strict result validation.
- Evaluation agent: new offline families; baseline/auto/routed comparison; selector stability and
  retrospective regret; label/test-feature invariance; independent review.
- Orchestrator: routed facade/cluster gate; integration; measured verification and model card.

## Frozen prospective evidence protocol
- Observed seeds 42,314,2027,4099 are development evidence.
- Seed 8081 is prospective confirmation, only after final production/config freeze; do not tune
  after inspecting it. Once inspected it is development for future changes.
- Main router comparison: nine new families, baseline/auto/routed, identical inputs and 50-second
  budgets, subprocess timeout60. Run development first and then fresh confirmation.
- Selector stability: bounded offline repeated splits; outer references remain harness-only.
- Categorical candidate stays disabled by default; use explicit experimental config/offline study.
- Near-tie tolerance defaults0; positive tolerances require development evidence, frozen config,
  and separate confirmation. No default behavior change without official alignment.
- Promotion requires no invalid output/new contract or leakage failures and reviewed tradeoffs;
  an inconclusive or negative comparison is reported, not hidden.

## Outstanding external gates
Official schemas/scoring/practice/checker and limits; clean official environment; largest-input
memory/deadline evidence; successful remote CI; public repository and submission metadata.

## Completed evidence
153 integrated tests passed, followed by eight strengthened post-prediction invariance checks.
148 offline development runs and 54 frozen CLI comparisons were valid; source/config unchanged.
Routed tied current auto quality in all 18 paired cases and added wall time for baseline-first
execution. Categorical default disabled, tie tolerance zero, baseline default retained.
Full findings: reports/router_validation.md. Prospective confirmation seed 8081 is now observed
and must be treated as development for future changes. Two subagents hit usage limits; parent
completed integration, diagnostics corrections, tests, benchmarks, documentation and model card.
