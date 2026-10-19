"""Review saved artifacts only; create a new directory and never overwrite history."""
import argparse
import hashlib
import json
from pathlib import Path

from promotion import evaluate

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory; historical outputs are immutable")
    current, current_hash = read("experiments/router-comparison.json")
    original, original_hash = read("experiments/baseline.json")
    alternate, alternate_hash=read("reports/grade-evaluation/accuracy-objective-development.json")
    registry = dict(schema_version=1,
        meaning="Read-only production incumbent plus provisional alternatives; not a claim of an objective-independent best model.",
        immutable_baseline=dict(path="experiments/baseline.json", report_sha256=original_hash,
                                source_sha256=original.get("config",{}).get("source_sha256")),
        tasks={t:dict(incumbent=dict(strategy="baseline",source_sha256=current["config"]["source_sha256"],
                    status="production_default_not_proven_score_optimal"),
                     frontier=[dict(strategy=s,source_sha256=current["config"]["source_sha256"]) for s in ("baseline","auto","routed")])
               for t in ("classification","regression","clustering")})
    registry["tasks"]["classification"]["frontier"].append(dict(strategy="routed",
        source_sha256=alternate["config"]["source_sha256"],config_sha256=alternate["config"]["router_config_sha256"],
        objective="accuracy",status="development_alternative_not_promoted"))
    policy = dict(status="provisional", task_metrics={
        "classification":{"accuracy":True,"balanced_accuracy":True,"macro_f1":True},
        "regression":{"rmse":False,"mae":False},
        "clustering":{"adjusted_rand_index":True,"adjusted_mutual_info":True}})
    specs = [
        ("routed-vs-baseline", "experiments/router-comparison.json", "baseline", "routed", ["baseline","auto","routed"]),
        ("routed-vs-auto", "experiments/router-comparison.json", "auto", "routed", ["baseline","auto","routed"]),
        ("accuracy-objective-vs-baseline", "reports/grade-evaluation/accuracy-objective-development.json", "baseline", "routed", ["baseline","routed"]),
        ("historical-auto-vs-baseline", "experiments/broad-comparison.json", "baseline", "auto", ["baseline","auto"]),
    ]
    outputs = {}
    for name, relative, baseline, candidate, pool in specs:
        report, digest = read(relative)
        expected = [{k:r[k] for k in ("task","name","seed")} for r in report["datasets"] if r["strategy"]==baseline]
        manifest=dict(stage="development", predeclared=False, previously_unopened=False,
                      inventory_origin="Retrospective inventory; cannot establish original coverage or fresh confirmation.",
                      report_sha256=digest, comparison_pool=pool, expected_cases=expected)
        assessed=evaluate(report,candidate=candidate,baseline=baseline,policy=policy,registry=registry,manifest=manifest)
        outputs[name]=dict(report=relative,report_sha256=digest,manifest=manifest,assessment=assessed)
    source_index={(r["task"],r["name"],r["seed"]):r for r in current["datasets"] if r["strategy"]=="routed"}
    cross=[]
    for row in alternate["datasets"]:
        if row["strategy"]!="routed": continue
        before=source_index[(row["task"],row["name"],row["seed"])]
        if before["data_sha256"]!=row["data_sha256"]:
            raise ValueError("Cross-objective input hash mismatch")
        cross.append(dict(name=row["name"],seed=row["seed"],data_sha256=row["data_sha256"],
            accuracy_change=row["metrics"]["accuracy"]-before["metrics"]["accuracy"],
            balanced_accuracy_change=row["metrics"]["balanced_accuracy"]-before["metrics"]["balanced_accuracy"],
            default_model=before["run_info"]["model"].get("selected_candidate"),
            accuracy_objective_model=row["run_info"]["model"].get("selected_candidate")))
    severe=[dict(name=c["name"],seed=c["seed"],metric=metric,change=c[metric],severity="severe_regression",
                 meaning="Investigation alert; not a hard objective or a grading verdict.")
            for c in cross for metric in ("accuracy_change","balanced_accuracy_change") if c[metric] <= -.10+1e-12]
    args.output_dir.mkdir(parents=True)
    write(args.output_dir/"registry.json",registry)
    write(args.output_dir/"objective-policy.json",policy)
    for name,result in outputs.items():
        write(args.output_dir/f"{name}.json",result)
    write(args.output_dir/"cross-objective.json",dict(default_report_sha256=current_hash,accuracy_report_sha256=alternate_hash,
        cases=cross, significant_regressions=severe,
        limitation="Same inputs across sessions; objective changes fitting weights and selection jointly; timing is not causal evidence."))
    summary={name:[dict(task=t["task"],outcome=t["outcome"],decision=t["development_decision"],
                        families=t["family_count"],missing_scenarios=t["missing_scenarios"],
                        severe_alerts=len(t["significant_regression_alerts"])) for t in result["assessment"]["task_assessments"]]
             for name,result in outputs.items()}
    write(args.output_dir/"summary.json",summary)
    print(json.dumps({"output_dir":str(args.output_dir),"comparisons":summary},indent=2))


if __name__=="__main__": main()
