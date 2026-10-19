# Independent review and improvement protocol

1. Three reviewers independently inspect data/profiling, execution/modelling, and evaluation.
   They record concrete reproductions and distinguish bugs from missing official information.
2. Resolve correctness findings and add regression tests before model experiments.
3. Freeze the original Extra Trees/K-means strategy as `baseline`; preserve the original report.
4. Broaden coverage to offline real datasets and several synthetic task families.
   Use seed 42 for development and seed 314 for confirmation. Confirmation reference labels
   are only consumed by the evaluation harness, never the candidate-selection code.
5. Independently implement supervised and clustering candidates. Their choices may use
   training-only validation or unsupervised diagnostics, never benchmark reference labels.
6. Compare baseline and candidate on exactly the same rows using actual CLI subprocesses.
   Every run has a timeout. Record validity, fallback use, family, seed, metrics, and wall time.
7. Report per-family wins and losses, not just a single average. Development metrics are
   provisional: balanced accuracy, regression RMSE/R2, and adjusted Rand index.
8. Promote defaults only when confirmation results support improvement without new failures;
   otherwise leave candidates opt-in and record limitations. This finite suite is not evidence
   of hidden-test superiority or worst-case runtime compliance.

The official checker, scoring formula, assessment limits, and practice tables remain absent.
Clean-environment installation and remote CI execution are separate outstanding checks.
