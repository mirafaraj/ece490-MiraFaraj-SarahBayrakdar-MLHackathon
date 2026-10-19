# Task-specific router implementation

Production source and `configs/router.json` are frozen for independent review and
development validation. Existing `baseline` and `auto` defaults are preserved.
The router is opt-in with `--strategy routed`; its objectives remain provisional.
Official metric alignment, environment verification, and fresh confirmation are
required before any default promotion.

## Decisions and metrics

The route plan uses training-only dimensions, feature types, missingness,
constant/duplicate columns, encoded-width estimates, and class counts. No dataset
name, benchmark seed, evaluation reference, or supervised prediction-feature value
is an input to routing or model selection.

Classification supports `balanced_accuracy` (provisional default, balanced class
weights) and `accuracy` (unweighted). Regression supports `rmse` (default) and
`mae`, both minimized in their natural positive scale. Clustering uses provisional
`bic`, with mixture eligibility gated by numeric feature shape and resource limits.

Eligibility limits bound optional candidates, dense allocation estimates, working
copies, sample size, and selection/refit time allocation. These estimates do not
certify whole-process maximum memory or execution time; the external worker
deadline and completed-baseline checkpoint provide the termination boundary.

Every eligible supervised candidate uses the same sampled fitting and validation
indices. Its preprocessing is independently fitted on the fitting rows. The winner
is refitted on full training rows before predicting. Lexical class labels and
fractional numeric class identities survive fitting, scoring, and prediction.

## Experimental category counts

The single-feature count classifier is disabled by default. It fits a class prior
and feature/category counts using fitting rows only:

`p(k|c) = (n(c,k) + alpha*prior[k]) / (n(c) + alpha)`.

Unseen categories use the fitting prior. Missing values and genuine strings use
disjoint token namespaces. Feature search follows schema order within configured
feature/cardinality/memory caps; the alpha grid has at most four values. Feature
and smoothing selection use the same inner validation rows as the other candidates.
Balanced accuracy uses prior-adjusted decisions while exposed probabilities remain
the normalized smoothed posterior.

## Strict partial configuration examples

Set the provisional classification objective to ordinary accuracy and enable the
category-count experiment, retaining the default total candidate cap of three:

```json
{
  "metrics": {"classification": "accuracy"},
  "experimental": {"categorical_enabled": true},
  "selection": {"near_tie_tolerance": 0.0}
}
```

The resulting eligible classification portfolio is normally `extra_trees`,
`categorical`, `linear`; histogram boosting is excluded by the candidate cap.
To compare all four, explicitly set `"limits": {"max_candidates": 4}`. Shape,
class, memory, and time gates can further reduce the portfolio.

```powershell
.venv312/Scripts/python.exe main.py --help
# Supply the contract's normal input/output options as well:
# ... --strategy routed --router-config configs/my_experiment.json
```

Unknown keys, unsupported metrics, nonfinite limits, invalid types, unsupported
policy versions, and malformed experimental settings fail validation. Repository
execution reads `configs/router.json`; an installed package without that repository
file uses identical versioned built-in defaults. Explicit missing paths fail.

The default tie tolerance is exactly zero: no unsupported positive tolerance was
chosen from observed outcomes. Within a configured absolute metric tolerance of
the globally best validation score, deterministic simplicity order is categorical,
linear, Extra Trees, then histogram boosting. This order is a policy assumption,
not a claim that one candidate is universally faster. Comparing to the globally
best score avoids chained near-ties drifting beyond tolerance.

## Verification

Command:

```powershell
.venv312/Scripts/python.exe -m pytest tests/test_routing.py tests/test_categorical.py tests/test_routed_supervised.py tests/test_supervised_candidates.py tests/test_routed_executor.py -q
```

Result: **43 passed**. Coverage includes strict configuration and JSON safety,
deterministic routing, dense/working-memory pruning, rare/low-budget cases,
metric-aware weights, shared fitting folds, test-feature selection invariance,
fractional and lexical class identities, hand-calculated smoothing, unseen/missing
categories, cardinality limits, categorical fold fitting and selected refit,
near-tie direction and drift prevention, existing auto behavior, and routed facade
baseline parity. This is focused local verification, not an official checker,
full-environment, performance, or promotion result.
