"""Dependency-free, artifact-only assessment; metrics are scenarios, not grades."""
from collections import defaultdict
import hashlib
import json
import math
import random
from statistics import mean

EPS = 1e-12
SCENARIOS = {
    "classification": {"accuracy": True, "balanced_accuracy": True, "macro_f1": True},
    "regression": {"rmse": False, "mae": False},
    "clustering": {"adjusted_rand_index": True, "adjusted_mutual_info": True},
}
EXPOSED_SEEDS = {42, 314, 2027, 4099, 8081}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def obj(value):
    return value if isinstance(value, dict) else {}


def valid_metric(metric, value):
    if not finite(value):
        return False
    if metric in {"accuracy", "balanced_accuracy", "macro_f1"}:
        return 0 <= value <= 1
    if metric in {"rmse", "mae"}:
        return value >= 0
    if metric in {"adjusted_rand_index", "adjusted_mutual_info"}:
        return -1 <= value <= 1
    return True  # Explicit external objective: direction and units belong to its policy.


def percentile(values, fraction):
    values = sorted(values)
    place = (len(values) - 1) * fraction
    low = int(place)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (place - low)


def family_interval(values, iterations=10000):
    """Family bootstrap, never treat row count or repeated seeds as independent N."""
    if len(values) < 2:
        return None
    rng = random.Random(490)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(iterations)]
    return [percentile(samples, .025), percentile(samples, .975)]


def quality_outcome(values):
    gain, loss = any(v > EPS for v in values), any(v < -EPS for v in values)
    return "tradeoff" if gain and loss else "harmful" if loss else "improvement" if gain else "equivalent"


def recovery(row):
    model = obj(obj(row.get("run_info")).get("model"))
    if not row.get("fallback"):
        return "none"
    if row.get("fallback_kind") == "completed_baseline" or model.get("fallback_kind") == "completed_baseline" or model.get("checkpoint_recovered"):
        return "completed_baseline"
    return "emergency"


def rank_regret(scores, strategy, higher):
    value = scores[strategy]
    better = sum((v > value + EPS) if higher else (v < value - EPS) for v in scores.values())
    tied = sum(abs(v - value) <= EPS for v in scores.values())
    return (better + (tied - 1) / 2) / max(1, len(scores) - 1)


def assess(report, candidate="auto", baseline="baseline", policy=None, registry=None,
           manifest=None, bootstrap_iterations=10000):
    if candidate == baseline:
        raise ValueError("Candidate must differ from reference")
    if not isinstance(bootstrap_iterations, int) or isinstance(bootstrap_iterations, bool) or bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations")
    report, policy, manifest = obj(report), obj(policy), obj(manifest)
    config = obj(report.get("config"))
    global_gaps, task_gaps, failures, alerts = [], defaultdict(list), [], []
    if not sha(config.get("source_sha256")) or config.get("source_unchanged") is not True or config.get("source_sha256_after") != config.get("source_sha256"):
        global_gaps.append("Completed, stable source/config fingerprints are required.")
    timeout = config.get("timeout_seconds")
    if not finite(timeout) or timeout <= 0:
        global_gaps.append("Finite positive external timeout is missing.")
    if config.get("router_config") and not sha(config.get("router_config_sha256")):
        global_gaps.append("Explicit router configuration hash is missing.")
    versions=config.get("strategy_versions")
    if versions is not None and (not isinstance(versions,dict) or any(not isinstance(v,dict) or not sha(v.get("source_sha256")) for v in versions.values())):
        global_gaps.append("Malformed frozen strategy-version provenance.")
    official = policy.get("status") == "official"
    if policy.get("status", "provisional") not in {"official", "provisional"}:
        raise ValueError("Objective policy status must be provisional or official")
    task_metrics = policy.get("task_metrics", {})
    if not isinstance(task_metrics, dict):
        raise ValueError("task_metrics must be an object")
    if official and (not policy.get("source") or not sha(policy.get("source_sha256"))):
        global_gaps.append("Official objective declaration needs a written source and its SHA-256.")
    if official and not task_metrics:
        global_gaps.append("Official objectives are declared but their scoring scenarios are missing.")
    metrics_by_task = {}
    for task, defaults in SCENARIOS.items():
        metrics = task_metrics.get(task, {} if official else defaults)
        if not isinstance(metrics, dict) or any(not isinstance(k, str) or not isinstance(v, bool) for k, v in metrics.items()):
            raise ValueError("Task metrics must map metric names to boolean higher-is-better directions")
        metrics_by_task[task] = metrics
    declared = config.get("strategies")
    if not isinstance(declared, list) or not all(isinstance(s, str) for s in declared) or not {baseline, candidate}.issubset(set(declared)):
        global_gaps.append("Strategy manifest is missing or does not include this comparison.")
        declared = [baseline, candidate]
    pool = manifest.get("comparison_pool", [baseline, candidate])
    if not isinstance(pool, list) or not all(isinstance(s, str) for s in pool) or len(set(pool)) != len(pool) or not {baseline, candidate}.issubset(set(pool)):
        raise ValueError("Comparison pool must contain unique reference/candidate strategy names")
    if not set(pool).issubset(set(declared)):
        global_gaps.append("Frozen comparison pool contains undeclared strategies.")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        global_gaps.append("A complete integer seed manifest is required.")
        seeds = []
    groups = defaultdict(dict)
    rows = report.get("datasets")
    if not isinstance(rows, list):
        global_gaps.append("Dataset rows are missing or malformed.")
        rows = []
    for row in rows:
        if not isinstance(row, dict):
            global_gaps.append("Malformed dataset row.")
            continue
        if not isinstance(row.get("strategy"), str) or row.get("strategy") not in pool:
            continue
        task, name, seed = row.get("task"), row.get("name"), row.get("seed")
        if not isinstance(task, str) or task not in SCENARIOS or not isinstance(name, str) or not name or not isinstance(seed, int) or isinstance(seed, bool):
            global_gaps.append("Malformed case identity or unsupported task.")
            continue
        key = (task, name, seed)
        if row["strategy"] in groups[key]:
            task_gaps[task].append(f"Duplicate strategy row: {name}/{seed}/{row['strategy']}")
        groups[key][row["strategy"]] = row
    for task, name in {(t, n) for t, n, _ in groups}:
        for seed in seeds:
            if (task, name, seed) not in groups:
                task_gaps[task].append(f"Missing declared seed/family pair: {name}/{seed}")
    expected = manifest.get("expected_cases")
    if expected is not None:
        if not isinstance(expected, list) or any(not isinstance(c, dict) or not isinstance(c.get("task"), str) or not isinstance(c.get("name"), str) or not isinstance(c.get("seed"), int) for c in expected):
            raise ValueError("expected_cases must contain task/name/seed objects")
        expected_keys = {(c["task"], c["name"], c["seed"]) for c in expected}
        if set(groups) != expected_keys or len(expected_keys) != len(expected):
            global_gaps.append("Observed case inventory differs from the predeclared manifest.")
    pairs = []
    for (task, name, seed), strategies in sorted(groups.items()):
        case = dict(task=task, name=name, seed=seed, family=name, issues=[], metrics={}, rank_regrets={})
        issues = case["issues"]
        if set(strategies) != set(pool):
            issues.append("Incomplete frozen comparison pool.")
        hashes = {r.get("data_sha256") for r in strategies.values() if isinstance(r.get("data_sha256"), str)}
        if len(hashes) != 1 or not all(sha(r.get("data_sha256")) for r in strategies.values()):
            issues.append("Missing or mismatched paired input hashes.")
        partitions = {r.get("partition") for r in strategies.values() if isinstance(r.get("partition"), str)}
        if len(partitions) != 1 or not all(isinstance(r.get("partition"), str) and r["partition"] for r in strategies.values()):
            issues.append("Missing or mismatched paired partition.")
        families = {r.get("family", name) for r in strategies.values() if isinstance(r.get("family", name), str)}
        if len(families) != 1 or not all(isinstance(r.get("family", name), str) and r.get("family", name) for r in strategies.values()):
            issues.append("Missing or mismatched family identity.")
        else:
            case["family"] = next(iter(families))
        for strategy, row in strategies.items():
            if row.get("valid") is not True or row.get("error") or row.get("exit_code") != 0:
                issues.append(f"{strategy}: invalid output or unsuccessful exit.")
                if strategy == candidate:
                    failures.append(dict(task=task, name=name, seed=seed, reason="Invalid candidate output or unsuccessful exit."))
            wall = row.get("wall_seconds")
            if not finite(wall) or wall < 0:
                issues.append(f"{strategy}: missing valid external wall time.")
            elif wall >= 60 or (finite(timeout) and wall >= timeout):
                issues.append(f"{strategy}: external deadline failed.")
                if strategy == candidate:
                    failures.append(dict(task=task, name=name, seed=seed, reason="Candidate external deadline failed."))
            if not isinstance(row.get("fallback"), bool):
                issues.append(f"{strategy}: fallback status is missing.")
        if issues:
            task_gaps[task].extend(f"{name}/{seed}: {issue}" for issue in issues)
            case["outcome"] = "inconclusive"
            pairs.append(case)
            continue
        before, after = strategies[baseline], strategies[candidate]
        case.update(data_sha256=after["data_sha256"], partition=after["partition"],
                    baseline_seconds=before["wall_seconds"], candidate_seconds=after["wall_seconds"],
                    baseline_recovery=recovery(before), candidate_recovery=recovery(after))
        if after["fallback"]:
            alerts.append(dict(task=task, name=name, seed=seed, severity="reliability_review",
                               reason="Completed baseline recovered" if recovery(after) == "completed_baseline" else "Emergency predictions require quality review"))
        diagnostic_metrics = {**SCENARIOS[task], **metrics_by_task[task]}
        for metric, higher in diagnostic_metrics.items():
            scores = {s: obj(row.get("metrics")).get(metric) for s, row in strategies.items()}
            for strategy,row in strategies.items():
                saved = obj(row.get("metrics"))
                if metric in saved and not valid_metric(metric,saved[metric]):
                    issue=f"{strategy}: invalid saved {metric} metric."
                    issues.append(issue)
                    task_gaps[task].append(f"{name}/{seed}: {issue}")
            if not all(valid_metric(metric, value) for value in scores.values()):
                continue
            b, a = scores[baseline], scores[candidate]
            raw = a - b if higher else b - a
            relative = (b - a) / b if metric in {"rmse", "mae"} and b > 0 else None
            case["metrics"][metric] = dict(baseline=b, candidate=a, raw_improvement=raw, relative_improvement=relative)
            if metric in metrics_by_task[task]:
                case["rank_regrets"][metric] = {s: rank_regret(scores, s, higher) for s in (baseline, candidate)}
            severe = raw <= -.10 + EPS if metric in {"accuracy", "balanced_accuracy", "macro_f1", "adjusted_rand_index", "adjusted_mutual_info"} else relative is not None and relative <= -.25 + EPS
            if severe:
                alerts.append(dict(task=task, name=name, seed=seed, metric=metric, severity="severe_regression",
                                   raw_improvement=raw, relative_improvement=relative,
                                   reason="Investigation alert, not a metric-specific promotion veto."))
        # Optional per-class recall evidence can expose harms hidden in aggregate metrics.
        recalls_b, recalls_a = obj(obj(before.get("metrics")).get("recall_by_class")), obj(obj(after.get("metrics")).get("recall_by_class"))
        for label in sorted(set(recalls_b) & set(recalls_a)):
            if finite(recalls_b[label]) and finite(recalls_a[label]) and recalls_a[label] - recalls_b[label] <= -.10 + EPS:
                alerts.append(dict(task=task, name=name, seed=seed, metric="recall_by_class", label=label,
                                   severity="severe_regression", raw_improvement=recalls_a[label]-recalls_b[label]))
        case["outcome"] = quality_outcome([v["raw_improvement"] for v in case["metrics"].values()]) if case["metrics"] else "inconclusive"
        pairs.append(case)
    task_results, advice = [], []
    for task in SCENARIOS:
        cases = [c for c in pairs if c["task"] == task]
        usable = [c for c in cases if not c["issues"]]
        metrics, missing = {}, []
        for metric in {**SCENARIOS[task], **metrics_by_task[task]}:
            measured = [c for c in usable if metric in c["metrics"]]
            if metric in metrics_by_task[task] and (len(measured) != len(cases) or not cases):
                missing.append(metric)
            if not measured:
                continue
            families = defaultdict(list)
            for case in measured:
                datum = case["metrics"][metric]
                families[case["family"]].append(datum["relative_improvement"] if metric in {"rmse", "mae"} else datum["raw_improvement"])
            valid_families = {f: mean(v for v in vals if v is not None) for f, vals in families.items() if any(v is not None for v in vals)}
            values = list(valid_families.values())
            raw = [c["metrics"][metric]["raw_improvement"] for c in measured]
            metrics[metric] = dict(family_weighted_improvement=mean(values) if values else None,
                interval95=family_interval(values, bootstrap_iterations), family_count=len(valid_families),
                units="relative loss reduction" if metric in {"rmse", "mae"} else "absolute metric improvement",
                zero_reference_cases=sum(c["metrics"][metric]["baseline"] == 0 for c in measured) if metric in {"rmse", "mae"} else 0,
                wins=sum(v > EPS for v in raw), ties=sum(abs(v) <= EPS for v in raw), losses=sum(v < -EPS for v in raw),
                worst_cases=sorted([dict(name=c["name"], seed=c["seed"], **c["metrics"][metric]) for c in measured], key=lambda c:c["raw_improvement"])[:5],
                family_values=valid_families)
        complete_metrics = [m for m in metrics_by_task[task] if all(m in c["rank_regrets"] for c in cases)] if cases and len(usable)==len(cases) else []
        family_ids = sorted({c["family"] for c in usable})
        by_scenario = {m: {f: {s: mean(c["rank_regrets"][m][s] for c in usable if c["family"] == f) for s in (baseline, candidate)} for f in family_ids} for m in complete_metrics}
        worst = {s: max((mean(vals[f][s] for f in family_ids) for vals in by_scenario.values()), default=None) for s in (baseline, candidate)}
        robust_delta = worst[baseline] - worst[candidate] if worst[baseline] is not None else None
        robust_ci = None
        if len(family_ids) >= 2 and by_scenario:
            rng = random.Random(490)
            samples = []
            for _ in range(bootstrap_iterations):
                sampled = rng.choices(family_ids, k=len(family_ids))
                rb = max(mean(vals[f][baseline] for f in sampled) for vals in by_scenario.values())
                ra = max(mean(vals[f][candidate] for f in sampled) for vals in by_scenario.values())
                samples.append(rb-ra)
            robust_ci = [percentile(samples, .025), percentile(samples, .975)]
        task_failures = [f for f in failures if f["task"] == task]
        severe_alerts = [a for a in alerts if a["task"] == task and a["severity"] == "severe_regression"]
        deltas = [c["metrics"][m]["raw_improvement"] for c in usable for m in c["metrics"]]
        outcome = quality_outcome(deltas) if deltas else "inconclusive"
        if global_gaps or task_gaps[task] or task_failures or not complete_metrics:
            outcome = "inconclusive"
        reg = obj(obj(registry).get("tasks"))
        incumbent = obj(obj(reg.get(task)).get("incumbent"))
        reference_version = obj(obj(config.get("strategy_versions")).get(baseline))
        reference_source = reference_version.get("source_sha256",config.get("source_sha256"))
        reference_strategy = reference_version.get("strategy",baseline)
        incumbent_match = incumbent.get("strategy") == reference_strategy and incumbent.get("source_sha256") == reference_source
        if incumbent.get("config_sha256") is not None:
            incumbent_match = incumbent_match and incumbent["config_sha256"] == reference_version.get("config_sha256",config.get("router_config_sha256"))
        policy_hash = hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        declared_fresh = manifest.get("stage") == "confirmation" and manifest.get("predeclared") is True and manifest.get("frozen_source_sha256") == config.get("source_sha256") and manifest.get("objective_policy_sha256") == policy_hash and manifest.get("previously_unopened") is True and expected is not None and not any(c["seed"] in EXPOSED_SEEDS for c in cases)
        decision = "blocked" if task_failures else "insufficient_evidence" if outcome == "inconclusive" else "retain_alternative"
        supported = robust_ci is not None and robust_ci[0] > EPS and len(family_ids) >= 8 and not missing
        if decision != "blocked" and outcome != "inconclusive" and supported:
            decision = "advance_provisionally"
            # Official promotion requires declared external attestations; never infer them from metric values.
            checks = obj(manifest.get("verified_checks"))
            if official and declared_fresh and incumbent_match and "comparison_pool" in manifest and all(checks.get(k) is True for k in ("official_checker", "profile_schema", "maximum_size", "memory_limit", "clean_environment", "immutable_baseline_compared", "fallback_quality")) and (not severe_alerts or manifest.get("severe_regressions_reviewed") is True):
                decision = "promote"
        times = [c["candidate_seconds"]-c["baseline_seconds"] for c in usable]
        measured_speed_gain = bool(times) and all(t <= EPS for t in times) and any(t < -EPS for t in times)
        recovery_difference = any(c["baseline_recovery"] != c["candidate_recovery"] for c in usable)
        if robust_ci is not None and robust_ci[1] < -EPS and decision not in {"blocked", "insufficient_evidence"} and not measured_speed_gain and not recovery_difference:
            decision = "reject"
        losses = [dict(name=c["name"], seed=c["seed"], metric=m, **v) for c in usable for m, v in c["metrics"].items() if v["raw_improvement"] < -EPS]
        item = dict(task=task, outcome=outcome, development_decision=decision, cases=len(cases), family_count=len(family_ids),
            metrics=metrics, missing_scenarios=missing, failures=task_failures, evidence_gaps=task_gaps[task],
            significant_regression_alerts=severe_alerts, incumbent_comparison="verified" if incumbent_match else "not_established",
            robust_ranking=dict(method="Worst metric-scenario mean fractional rank regret; equal family weights; not official points.",
                comparison_pool=pool, pool_frozen_by_manifest="comparison_pool" in manifest, worst_regret=worst,
                improvement=robust_delta, interval95=robust_ci, bootstrap_unit="family", bootstrap_iterations=bootstrap_iterations,
                limited_independence="Repeated seeds/splits are averaged within families; mechanisms may still be related."),
            mean_extra_wall_seconds=mean(times) if times else None,
            observed_cost_verdict="dominated_on_measured_quality_and_time" if times and outcome == "equivalent" and all(t >= -EPS for t in times) and any(t > EPS for t in times) else "no_dominance_established")
        task_results.append(item)
        if cases:
            actions = {
                "classification": "Separate class-weight policy from estimator selection; score the same fitted candidates under all scenarios and inspect class recall.",
                "regression": "Preserve linear-family gains; examine missing/product and heavy-tail failures with equal-budget repeated training-only holdouts.",
                "clustering": "Reproduce losing missing-axis cases; ablate indicators and covariance eligibility separately, retaining complete-data controls.",
            }
            advice.append(dict(priority=1 if losses or severe_alerts else 2, task=task, action=actions[task], evidence=losses[:10],
                severe_alert_count=len(severe_alerts), experiment="Change one factor; pair frozen candidates on identical inputs; reserve unopened families for confirmation.",
                acceptance="Improve official score when available, otherwise improve robust scenario ranking with family-level uncertainty; no metric-specific zero-loss veto."))
            if times and mean(times) > EPS:
                advice.append(dict(priority=2, task=task, action="Measure checkpoint-first overhead and recovery quality at 10/25/50-second budgets before changing safeguards.", mean_extra_wall_seconds=mean(times)))
    gaps = global_gaps + [f"{t}: {g}" for t in SCENARIOS for g in task_gaps[t]]
    return dict(assessment_mode="scoring_uncertainty", schema_version=2, candidate_strategy=candidate, baseline_strategy=baseline,
        objective_policy=dict(status="official_declared" if official else "provisional", source=policy.get("source"),
            meaning="No official points estimated; metric scenarios express uncertainty, not protected objectives.", metrics=metrics_by_task),
        reliability=dict(decision="blocked" if failures else "insufficient_evidence" if gaps else "review_required" if any(a["severity"] == "reliability_review" for a in alerts) else "passed", failures=failures, gaps=gaps),
        candidate_failures=failures, evidence_gaps=gaps, cases=pairs, task_assessments=task_results,
        significant_regressions=alerts, ranked_advice=sorted(advice, key=lambda a:a["priority"]),
        promotion=dict(decision="blocked" if failures else "insufficient_evidence", task_decisions={t["task"]:t["development_decision"] for t in task_results},
            required=["Declared official scoring/normalization", "Verified incumbent and immutable baseline comparison", "Fresh frozen confirmation", "Official contract/profile/environment/resource checks"],
            hidden_set_guarantee=False),
        limitations=["Historical predictions, macro F1, MAE, AMI and class recalls cannot be reconstructed unless supplied.",
                    "No expected total score can be calculated without official formulas or defensible dataset-distribution weights.",
                    "Rank regret ignores effect magnitude; worst-case deltas and severe-regression review remain essential.",
                    "Missing scenarios and fewer than eight family groups preclude promotion; historical confirmation labels do not prove freshness.",
                    "No case manifest means entirely omitted families cannot be detected."])
