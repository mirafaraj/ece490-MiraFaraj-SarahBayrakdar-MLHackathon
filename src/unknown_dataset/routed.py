"""Execute an explicit task policy while preserving completed baseline predictions."""

from __future__ import annotations

import time


def fit_predict_routed(task, X_train, y_train=None, X_test=None, feature_types=None,
                       n_clusters=None, time_budget=50.0, seed=42, route_config=None,
                       checkpoint_callback=None):
    """Build a training-only plan, checkpoint its baseline, then try eligible work.

    The outer worker owns the hard deadline. A checkpoint contains complete ordered
    predictions; it does not contain a partial estimator or a validation-only fit.
    """
    started = time.monotonic()
    from .routing import summarize_training, make_route_plan

    summary = summarize_training(task, X_train, y_train, feature_types)
    plan = make_route_plan(summary, time_budget=time_budget, n_clusters=n_clusters,
                           config=route_config)
    common = dict(task=task, X_train=X_train, y_train=y_train, X_test=X_test,
                  feature_types=feature_types, n_clusters=n_clusters, seed=seed)
    if task == "clustering":
        from .models import fit_predict

        predictions, baseline_info = fit_predict(**common, time_budget=time_budget, strategy="baseline")
    else:
        from .supervised import fit_baseline_supervised

        predictions, baseline_info = fit_baseline_supervised(
            **common, time_budget=time_budget, route_plan=plan)

    baseline_info = dict(baseline_info)
    baseline_info.update(strategy="routed", routing=plan.to_dict(), fallback=False,
                         route_stage="completed_baseline", checkpoint_recovered=False)
    if checkpoint_callback is not None:
        checkpoint_callback(predictions, baseline_info)
    baseline_elapsed = time.monotonic() - started
    # Recompute deadline headroom after full-data fitting and checkpoint writing.
    remaining = time_budget - baseline_elapsed
    optional = [name for name in plan.candidates if name not in {"extra_trees", "kmeans"}]
    minimum_optional = max(4.0, float(getattr(plan, "reserve_seconds", 3.0)))
    if not optional or remaining < minimum_optional:
        baseline_info.update(route_stage="baseline_only", model_seconds=time.monotonic() - started,
                             optional_skipped="no eligible optional candidates" if not optional
                             else "insufficient deadline headroom after baseline")
        return predictions, baseline_info

    try:
        if task == "clustering":
            from .clustering import fit_predict_clustering

            replacement, info = fit_predict_clustering(**common, time_budget=remaining,
                                                        route_plan=plan)
        else:
            from .supervised import fit_predict_supervised

            replacement, info = fit_predict_supervised(**common, time_budget=remaining,
                                                       route_plan=plan,
                                                       completed_baseline=(predictions, baseline_info))
        info = dict(info)
        refit_failure = info.get("refit_fallback_reason")
        info.update(strategy="routed", routing=plan.to_dict(), route_stage="completed_selection",
                    baseline_model=baseline_info.get("model"), baseline_seconds=baseline_elapsed,
                    model_seconds=time.monotonic() - started, checkpoint_recovered=False,
                    fallback=bool(refit_failure))
        if refit_failure:
            info.update(fallback_kind="completed_baseline", fallback_reason=refit_failure,
                        route_stage="optional_refit_error")
        return replacement, info
    except Exception as exc:
        # Optional work must not destroy already completed predictions. Keep the
        # cause visible in run metadata so this is distinguishable from a normal route.
        baseline_info.update(route_stage="optional_error", fallback=True,
                             fallback_kind="completed_baseline", checkpoint_recovered=False,
                             fallback_reason=f"{type(exc).__name__}: {exc}",
                             model_seconds=time.monotonic() - started)
        return predictions, baseline_info
