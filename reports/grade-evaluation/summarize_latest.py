"""Independently reproduce completed router summaries and offline selector diagnostics.

Artifact reads only. Run from the repository root; outputs only in this directory.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def load(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def verify_summary(raw, summary, classification_metric="balanced_accuracy"):
    assert raw["config"]["source_unchanged"] is True
    assert raw["config"]["source_sha256"] == raw["config"]["source_sha256_after"] == summary["source_sha256"]
    baseline, candidate = summary["reference_strategy"], summary["candidate_strategy"]
    grouped = defaultdict(dict)
    for row in raw["datasets"]:
        if row["strategy"] in {baseline,candidate}:
            grouped[(row["seed"],row["task"],row["name"])][row["strategy"]] = row
    independent = []
    for (seed,task,name), pair in grouped.items():
        b,a = pair[baseline],pair[candidate]
        assert b["data_sha256"] == a["data_sha256"] and b["valid"] and a["valid"]
        metric = {"classification":classification_metric,"regression":"rmse","clustering":"adjusted_rand_index"}[task]
        delta = a["metrics"][metric]-b["metrics"][metric]
        delta = -delta if task == "regression" else delta
        independent.append(dict(seed=seed,task=task,name=name,partition=b["partition"],change=delta,
            relative=1-a["metrics"][metric]/b["metrics"][metric] if task == "regression" else None,
            outcome="tie" if abs(delta)<=1e-9 else "win" if delta>0 else "loss",
            before=b["wall_seconds"],after=a["wall_seconds"]))
    checks=0
    for aggregate in summary["aggregates"]:
        rows=[r for r in independent if (r["partition"],r["task"])==(aggregate["partition"],aggregate["task"])]
        expected={"cases":len(rows),"outcomes":{x:sum(r["outcome"]==x for r in rows) for x in ("win","tie","loss","invalid")},
            f"mean_{baseline}_seconds":mean(r["before"] for r in rows),f"mean_{candidate}_seconds":mean(r["after"] for r in rows),
            "mean_quality_change":None if aggregate["task"]=="regression" else mean(r["change"] for r in rows),
            "mean_relative_rmse_reduction":mean(r["relative"] for r in rows) if aggregate["task"]=="regression" else None}
        for key,value in expected.items():
            actual=aggregate[key]
            assert math.isclose(value,actual,abs_tol=1e-12,rel_tol=1e-10) if isinstance(value,float) else value==actual,(key,value,actual)
            checks+=1
    return dict(reference=baseline,candidate=candidate,classification_metric=classification_metric,pairs=len(independent),aggregate_fields_verified=checks)


def main():
    raw,digest=load(ROOT/"experiments/router-comparison.json")
    validations=[]
    for filename,metric in (("router-vs-baseline.json","balanced_accuracy"),("router-vs-baseline-accuracy.json","accuracy"),("router-vs-auto.json","balanced_accuracy")):
        summary,_=load(ROOT/"experiments"/filename)
        validations.append(dict(file=filename,**verify_summary(raw,summary,metric)))
    strategies={}
    for strategy in ("baseline","auto","routed"):
        rows=[r for r in raw["datasets"] if r["strategy"]==strategy]
        strategies[strategy]=dict(runs=len(rows),valid=sum(r["valid"] for r in rows),fallbacks=sum(r.get("fallback",False) for r in rows),
            mean_wall_seconds=mean(r["wall_seconds"] for r in rows),max_wall_seconds=max(r["wall_seconds"] for r in rows),
            max_training_rows=max(r["run_info"]["model"]["training_rows"] for r in rows),
            max_input_features=max(r["run_info"]["model"]["input_features"] for r in rows))
    offline,offline_digest=load(ROOT/"experiments/selector_stability.json")
    assert offline["config"]["source_unchanged"] is True
    assert offline["config"]["source_sha256"]==offline["config"]["source_sha256_after"]
    groups=defaultdict(dict)
    for row in offline["runs"]:
        groups[(row["name"],row["split_seed"],row["task"])][row["strategy"]]=row
    comparisons={}
    for reference,candidate in (("baseline","routed"),("routed","unrestricted"),("routed","routed_tie"),("routed","routed_categorical")):
        rows=[]
        for (name,seed,task),pair in groups.items():
            if {reference,candidate}.issubset(pair):
                b,a=pair[reference],pair[candidate]
                assert b["data_sha256"]==a["data_sha256"] and b["valid"] and a["valid"]
                delta=a["score"]-b["score"] # offline scores explicitly higher-is-better
                rows.append(dict(name=name,split_seed=seed,task=task,quality_change=delta,
                    outcome="tie" if abs(delta)<=1e-12 else "win" if delta>0 else "loss"))
        comparisons[f"{candidate}_vs_{reference}"]=dict(cases=len(rows),outcomes=dict(Counter(r["outcome"] for r in rows)),
            nonties=[r for r in rows if r["outcome"]!="tie"])
    result=dict(notice="Artifact-derived diagnostics only. No official points. Offline 120s direct calls are not 60s CLI compliance. Seeds42/314/8081 exposed for future tuning.",
        cli_report_sha256=digest,source_sha256=raw["config"]["source_sha256"],summary_validation=validations,cli_strategies=strategies,
        offline=dict(report_sha256=offline_digest,runs=len(offline["runs"]),valid=sum(r["valid"] for r in offline["runs"]),
            time_budget=offline["config"]["time_budget"],comparisons=comparisons,
            candidate_selection_frequencies={strategy:dict(Counter(r["selected_candidate"] for r in offline["runs"] if r["strategy"]==strategy)) for strategy in ("auto","routed","unrestricted","routed_tie")}))
    objective_path=OUT/"accuracy-objective-development.json"
    if objective_path.exists():
        alternative,alternative_digest=load(objective_path)
        assert alternative["config"]["source_unchanged"] is True
        assert alternative["config"]["source_sha256"]==alternative["config"]["source_sha256_after"]
        override=Path(alternative["config"]["router_config"])
        digest=hashlib.sha256()
        for path in sorted([ROOT/"main.py",*ROOT.glob("src/**/*.py"),*ROOT.glob("scripts/*.py"),*ROOT.glob("configs/*.json")]):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
        assert digest.hexdigest()==raw["config"]["source_sha256"],"Current production differs from frozen source"
        digest.update(b"explicit-router-config")
        digest.update(override.read_bytes())
        assert digest.hexdigest()==alternative["config"]["source_sha256"],"Explicit objective config does not explain source fingerprint"
        assert hashlib.sha256(override.read_bytes()).hexdigest()==alternative["config"]["router_config_sha256"]
        earlier={(r["seed"],r["task"],r["name"]):r for r in raw["datasets"] if r["strategy"]=="routed" and r["task"]=="classification"}
        paired=[]
        for a in alternative["datasets"]:
            if a["strategy"]!="routed":
                continue
            b=earlier[(a["seed"],a["task"],a["name"])]
            assert a["data_sha256"]==b["data_sha256"] and a["valid"] and b["valid"]
            paired.append(dict(name=a["name"],seed=a["seed"],
                default_accuracy=b["metrics"]["accuracy"],accuracy_objective_accuracy=a["metrics"]["accuracy"],
                accuracy_change=a["metrics"]["accuracy"]-b["metrics"]["accuracy"],
                default_balanced_accuracy=b["metrics"]["balanced_accuracy"],accuracy_objective_balanced_accuracy=a["metrics"]["balanced_accuracy"],
                balanced_accuracy_change=a["metrics"]["balanced_accuracy"]-b["metrics"]["balanced_accuracy"]))
        result["objective_variant_comparison"]=dict(report_sha256=alternative_digest,source_sha256=alternative["config"]["source_sha256"],
            production_and_override_fingerprints_reproduced=True,matched_cases=len(paired),cases=paired,
            mean_accuracy_change=mean(p["accuracy_change"] for p in paired),
            mean_balanced_accuracy_change=mean(p["balanced_accuracy_change"] for p in paired),
            outcome="tradeoff",timing_notice="Cross-session timing is observational; not causal evidence of objective configuration cost.",
            development_only=True)
    (OUT/"latest-independent-evidence.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
