# Fixed classification resource rule — registered before validation

Branch: codex/classification-runtime-repair. Parent committed state:
5b12dc2d54f0b0e2807ee823630b61a9ed879b5a, retaining the six uncommitted integration files.
No threshold sweep or repeat repair cycle is authorized.

Keep all existing raw-size guards. Additionally, for classification under baseline,
auto or submission, activate the existing bounded-forest -> linear recovery path when:

- Full training rows > 20,000 (the existing fitting-row cap).
- At least one categorical feature is declared/inferred from training data.
- Full rows * prepared-width upper bound > 2,000,000 (the existing dense cell cap).

Prepared-width upper bound, computed from full training features only:
numeric columns contribute one value plus one indicator if cleaning creates missing values;
categorical columns contribute min(32, distinct cleaned nonmissing strings + missing-token presence),
at least one. Constants and duplicates are conservatively counted, so this is an upper bound,
not a replacement for the actual fitted width or a new preprocessing implementation.
No evaluation features, labels, dataset names or saved scores enter admission.

For admitted tables, skip unbounded portfolio selection/refits: use the existing 16-tree,
sqrt-feature, max-depth-24 forest on a deterministic class-preserving sample of at most
20,000 rows. If that attempt fails, use the existing linear learned recovery. The existing
single absolute parent deadline, <=25s/65%-available forest allowance, eight-second output/
recovery reserve and final emergency recovery remain unchanged. No cap or deadline increases.
The CLI retains the same 50-second internal budget and must finish below 45 seconds in validation.
The ordinary Python API retains its caller-owned hard-deadline contract.

This admits feasible full-training work before a sampled histogram winner can lead to an
infeasible dense refit and then an unrestricted forest. Existing estimators, weights, metrics,
preprocessing and ordinary-table selection are unchanged. Numeric-only and regression/clustering
paths do not gain a new rule. Routed retains its existing parent admission rule.
This threshold is an engineering assumption, not an organizer requirement or universal guarantee.

Validation order, frozen before fitting:
1. Exact saved 45k x50 fixture, explicit auto then baseline, once each, seed 47131.
   Independent complete profiles; valid learned output, no emergency, full-process <45s.
   Record accuracy, balanced accuracy, macro F1, each class recall and training-majority comparator.
   Require accuracy, balanced accuracy and macro F1 each above that majority comparator;
   valid output alone does not establish predictive usefulness.
   Stop if either fails; do not revise this preset or run a second repair attempt.
2. Only after both pass: repeat each once; focused boundaries/recovery tests; remaining frozen
   integration probes; 20 classification +18 regression/clustering historical comparisons;
   explicit-strategy/default comparisons; all 39 original stress strategy cases; full suite.
3. Any normal-size prediction difference stops the task. For successful stress cases, report
   any prediction changes and quality losses explicitly. Finishing alone is insufficient evidence.
4. Reuse completed results only if relevant frozen source hashes still match. Retain all previous
   failures, runs and manifests. Keep new raw outputs local, separate from historical evidence.

No staging, commit, merge, push, default adjustment or main modification is authorized.
