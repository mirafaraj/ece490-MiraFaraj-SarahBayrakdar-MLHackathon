"""Task-aware artifact assessment with optional strict ordinary-accuracy diagnostic."""
from collections import defaultdict
import math
from statistics import mean

TOLERANCE = 1e-12
PRIMARY = {"classification": "accuracy", "regression": "rmse", "clustering": "adjusted_rand_index"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def fingerprint(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def strict_evaluate(report, candidate="auto", baseline="baseline"):
    failures, gaps, pairs = [], [], []
    if candidate == baseline:
        raise ValueError("Candidate must differ from baseline")
    if not isinstance(report, dict):
        report = {}
    config = report.get("config", {})
    if not isinstance(config, dict):
        config = {}
    if not fingerprint(config.get("source_sha256")):
        gaps.append("Missing or malformed frozen source SHA-256.")
    if config.get("source_unchanged") is not True:
        gaps.append("Source stability is absent or false; completed frozen evidence required.")
    if config.get("source_sha256_after") != config.get("source_sha256"):
        gaps.append("Source before/after fingerprints do not match.")
    timeout = config.get("timeout_seconds")
    if not finite(timeout) or timeout <= 0:
        gaps.append("Missing finite positive external timeout.")
    groups = defaultdict(dict)
    rows = report.get("datasets", [])
    if not isinstance(rows, list):
        rows = []
        gaps.append("Dataset rows are not a list.")
    for row in rows:
        if not isinstance(row, dict):
            gaps.append("Malformed dataset row.")
            continue
        if row.get("strategy") not in {baseline, candidate}:
            continue
        key = (row.get("seed"), row.get("task"), row.get("name"))
        if row.get("strategy") in groups[key]:
            gaps.append(f"Duplicate strategy row: {key}/{row.get('strategy')}")
        groups[key][row.get("strategy")] = row
    for key, strategies in groups.items():
        seed, task, name = key
        item = dict(seed=seed, task=task, name=name, decision="insufficient_evidence", issues=[])
        issues = item["issues"]
        if None in key or task not in PRIMARY:
            issues.append("Missing case identity or unsupported task.")
        if set(strategies) != {baseline, candidate}:
            issues.append("Missing baseline or candidate pair.")
            gaps.append(f"{key}: incomplete paired evidence.")
            pairs.append(item)
            continue
        before, after = strategies[baseline], strategies[candidate]
        item["baseline_fallback"] = before.get("fallback")
        item["candidate_fallback"] = after.get("fallback")
        model = after.get("run_info", {}).get("model", {}) if isinstance(after.get("run_info"), dict) else {}
        item["candidate_recovery_metadata"] = {k: v for k, v in model.items() if "fallback" in k or "recovery" in k or "baseline" in k}
        # A known valid matched-input accuracy violation remains a rejection even
        # when unrelated metrics or runtime/provenance metadata are incomplete.
        bm = before.get("metrics") if isinstance(before.get("metrics"), dict) else {}
        am = after.get("metrics") if isinstance(after.get("metrics"), dict) else {}
        if (task == "classification" and before.get("valid") is True and after.get("valid") is True
                and fingerprint(before.get("data_sha256")) and before.get("data_sha256") == after.get("data_sha256")
                and all(finite(v) and 0 <= v <= 1 for v in (bm.get("accuracy"), am.get("accuracy")))):
            delta = am["accuracy"] - bm["accuracy"]
            item["accuracy_change"] = delta
            item["balanced_accuracy_change"] = am["balanced_accuracy"] - bm["balanced_accuracy"] if all(finite(v) for v in (bm.get("balanced_accuracy"), am.get("balanced_accuracy"))) else None
            if delta < -TOLERANCE:
                failures.append(dict(case=list(key), reason="Ordinary accuracy decreases.", accuracy_change=delta))
        if not fingerprint(before.get("data_sha256")) or before.get("data_sha256") != after.get("data_sha256"):
            issues.append("Missing/mismatched paired input fingerprints.")
        if not before.get("partition") or before.get("partition") != after.get("partition"):
            issues.append("Missing/mismatched paired partition.")
        metrics = [PRIMARY.get(task)] + (["balanced_accuracy"] if task == "classification" else [])
        for label, row in ((baseline, before), (candidate, after)):
            if row.get("valid") is not True:
                failures.append(dict(case=list(key), reason=f"{label} output invalid or validity absent."))
                issues.append(f"{label} invalid output.")
            if not isinstance(row.get("fallback"), bool):
                issues.append(f"{label} fallback status absent.")
            if row.get("error"):
                failures.append(dict(case=list(key), reason=f"{label} recorded an error."))
                issues.append(f"{label} recorded an error.")
            if row.get("exit_code") != 0:
                issues.append(f"{label} successful exit status absent.")
            if not finite(row.get("wall_seconds")) or row.get("wall_seconds", -1) < 0:
                issues.append(f"{label} finite nonnegative wall time absent.")
            elif finite(timeout) and row["wall_seconds"] >= timeout:
                failures.append(dict(case=list(key), reason=f"{label} meets/exceeds external timeout."))
                issues.append(f"{label} external deadline failed.")
            for metric in metrics:
                row_metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                value = row_metrics.get(metric)
                if not finite(value):
                    issues.append(f"{label} missing/nonfinite {metric}.")
                elif metric in {"accuracy", "balanced_accuracy"} and not 0 <= value <= 1:
                    issues.append(f"{label} {metric} outside [0,1].")
                elif metric == "rmse" and value < 0:
                    issues.append(f"{label} negative RMSE.")
                elif metric == "adjusted_rand_index" and not -1 <= value <= 1:
                    issues.append(f"{label} ARI outside [-1,1].")
        if after.get("fallback") is True and before.get("fallback") is False:
            failures.append(dict(case=list(key), reason="Candidate introduces a fallback."))
            issues.append("Candidate fallback regression.")
        elif after.get("fallback") is True:
            issues.append("Both strategies fallback; useful-quality evidence is insufficient.")
        if issues:
            gaps.extend(f"{key}: {issue}" for issue in issues)
        else:
            item["decision"] = "passed"
            metric = PRIMARY[task]
            b, a = before["metrics"][metric], after["metrics"][metric]
            item.update(metric=metric, baseline=b, candidate=a, quality_change=b-a if task == "regression" else a-b,
                        baseline_seconds=before["wall_seconds"], candidate_seconds=after["wall_seconds"])
            if task == "classification":
                delta = a-b
                item.update(accuracy_change=delta,
                    balanced_accuracy_change=after["metrics"]["balanced_accuracy"]-before["metrics"]["balanced_accuracy"])
                if delta < -TOLERANCE:
                    item["decision"] = "rejected"
            elif task == "regression":
                item["relative_rmse_reduction"] = 1-a/b if b else None
            # Passing non-classification rows means usable evidence, not quality promotion.
        if any(f["case"] == list(key) for f in failures):
            item["decision"] = "rejected"
        pairs.append(item)
    classification = [p for p in pairs if p["task"] == "classification"]
    measured = [p for p in classification if "accuracy_change" in p]
    if not classification:
        gaps.append("No classification coverage; ordinary-accuracy gate cannot pass.")
    if not groups:
        gaps.append("No paired baseline/candidate evidence.")
    if not config.get("strategies") or not {baseline, candidate}.issubset(set(config["strategies"])):
        gaps.append("Report strategy manifest does not include requested comparison.")
    declared_seeds = config.get("seeds")
    if not isinstance(declared_seeds, list) or not declared_seeds:
        gaps.append("Missing seed manifest; completeness cannot be established.")
    else:
        families_seen = {(p["task"], p["name"]) for p in pairs}
        observed = {(p["seed"], p["task"], p["name"]) for p in pairs}
        for seed in declared_seeds:
            for task, name in families_seen:
                if (seed, task, name) not in observed:
                    gaps.append(f"Missing declared seed/family pair: {seed}/{task}/{name}.")
    family_scores = []
    for name in sorted({p["name"] for p in classification}, key=str):
        family = [p for p in classification if p["name"] == name]
        usable = [p for p in family if "accuracy_change" in p]
        family_scores.append(dict(name=name, cases=len(family), measured_cases=len(usable),
            mean_accuracy_change=mean(p["accuracy_change"] for p in usable) if usable else None,
            worst_accuracy_change=min((p["accuracy_change"] for p in usable), default=None),
            mean_balanced_accuracy_change=mean(p["balanced_accuracy_change"] for p in usable if finite(p.get("balanced_accuracy_change"))) if any(finite(p.get("balanced_accuracy_change")) for p in usable) else None,
            decision="rejected" if any(p["decision"] == "rejected" for p in family) else "insufficient_evidence" if any(p["decision"] != "passed" for p in family) else "passed"))
    decision = "rejected" if failures else "insufficient_evidence" if gaps else "passed"
    missing_tasks = sorted(set(PRIMARY) - {p["task"] for p in pairs})
    advice = []
    accuracy_losses = [p for p in measured if p["accuracy_change"] < -TOLERANCE]
    if accuracy_losses:
        advice.append(dict(priority=1, action="Reject this candidate for accuracy-preserving promotion; retain the baseline classification path. Evaluate unweighted candidate ablations and select by ordinary accuracy on training-only validation.",
            evidence=[dict(name=p["name"], seed=p["seed"], accuracy_change=p["accuracy_change"], balanced_accuracy_change=p["balanced_accuracy_change"]) for p in accuracy_losses]))
    if failures or gaps:
        advice.append(dict(priority=2, action="Resolve invalid, fallback, provenance, timing, or metric gaps and rerun a completed frozen comparison before claiming a gate pass.", failures=len(failures), evidence_gaps=len(gaps)))
    for task in ("clustering", "regression"):
        usable = [p for p in pairs if p["task"] == task and "quality_change" in p]
        weak = [p for p in usable if p["quality_change"] <= TOLERANCE]
        if weak:
            advice.append(dict(priority=3 if task == "clustering" else 4,
                action="Target mixed/curved clustering representation and safe routing, keeping the ongoing GMM fix separate." if task == "clustering" else "Investigate mixed/missing regression and holdout stability; do not hide family losses behind linear-family gains.",
                evidence=[dict(name=p["name"], seed=p["seed"], baseline=p["baseline"], candidate=p["candidate"], quality_change=p["quality_change"]) for p in weak]))
    advice.append(dict(priority=5, action="Freeze candidate/configuration, align official scoring and profile schema, then evaluate fresh unopened confirmation and maximum-size cases on assessment hardware."))
    return dict(candidate_strategy=candidate, baseline_strategy=baseline, tolerance=TOLERANCE,
        measured_accuracy_gate=dict(decision=decision, scope="Observed paired cases only; any single measured ordinary-accuracy loss rejects, even if the mean improves.",
            classification_cases=len(classification), measured_classification_cases=len(measured),
            mean_accuracy_change=mean(p["accuracy_change"] for p in measured) if measured else None,
            worst_accuracy_change=min((p["accuracy_change"] for p in measured), default=None),
            mean_balanced_accuracy_change=mean(p["balanced_accuracy_change"] for p in measured if finite(p.get("balanced_accuracy_change"))) if any(finite(p.get("balanced_accuracy_change")) for p in measured) else None),
        promotion=dict(decision="rejected" if decision == "rejected" else "insufficient_evidence",
            required=["Official metric/profile alignment", "Fresh unopened confirmation after candidate freeze", "Official maximum-size/runtime/memory coverage"],
            hidden_set_guarantee=False),
        candidate_failures=failures, evidence_gaps=sorted(set(gaps)),
        missing_coverage=dict(tasks=missing_tasks, official_datasets="Not established by this artifact", maximum_size="Not established by this artifact", fresh_confirmation="Not established by this artifact"),
        classification_families=family_scores, cases=pairs, ranked_advice=advice)



def evaluate(report, candidate="auto", baseline="baseline", classification_objective=None,
             secondary_loss_budgets=None, policy=None, registry=None, manifest=None,
             bootstrap_iterations=10000):
    """Neutral promotion assessment; former metric budgets are diagnostic metadata only."""
    from assessment import assess
    if classification_objective not in {None, "accuracy", "balanced_accuracy"}:
        raise ValueError("Unknown legacy classification diagnostic")
    if secondary_loss_budgets is not None:
        if not isinstance(secondary_loss_budgets, dict) or set(secondary_loss_budgets)-{"accuracy", "balanced_accuracy"}:
            raise ValueError("Unknown legacy diagnostic loss budget")
        if any(not finite(v) or v < 0 for v in secondary_loss_budgets.values()):
            raise ValueError("Diagnostic budgets must be finite and nonnegative")
    result = assess(report, candidate, baseline, policy, registry, manifest, bootstrap_iterations)
    try:
        diagnostic = strict_evaluate(report, candidate, baseline)
        result["measured_accuracy_gate"] = diagnostic["measured_accuracy_gate"]
    except (AttributeError, KeyError, TypeError, ValueError):
        result["measured_accuracy_gate"] = {"decision":"insufficient_evidence"}
    result["strict_accuracy_diagnostic"] = {
        "measured_gate":result["measured_accuracy_gate"],
        "affects_promotion":False,
        "meaning":"Optional historical no-drop diagnostic; never an optimization objective."}
    result["legacy_diagnostics"] = {"classification_objective":classification_objective,
        "secondary_loss_budgets":secondary_loss_budgets, "affects_promotion":False}
    return result
