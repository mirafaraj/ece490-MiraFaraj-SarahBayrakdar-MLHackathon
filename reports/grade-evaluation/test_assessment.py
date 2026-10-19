"""Evaluator policy tests only: no ML imports, data generation, or fitting."""
from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import analyze
from assessment import assess, family_interval
from prediction_evidence import enrich, metrics
from reference_evidence import pair_reference
from promotion import evaluate
from test_promotion import fixture


def task(result, name="classification"):
    return next(t for t in result["task_assessments"] if t["task"] == name)


def many_families(seed=99173):
    report = fixture()
    report["config"]["seeds"] = [seed]
    report["datasets"] = []
    for i in range(8):
        rows = fixture()["datasets"]
        for row in rows:
            row.update(name=f"family{i}", family=f"family{i}", seed=seed)
            row["metrics"]["macro_f1"] = .7
        rows[1]["metrics"] = {"accuracy":.9,"balanced_accuracy":.8,"macro_f1":.8}
        report["datasets"].extend(rows)
    return report


class ScenarioAssessmentTests(unittest.TestCase):
    def test_no_implicit_primary_metric_or_loss_veto(self):
        report = fixture()
        report["datasets"][1]["metrics"] = {"accuracy":.6,"balanced_accuracy":.9}
        result = evaluate(report,bootstrap_iterations=100)
        self.assertEqual(task(result)["outcome"],"tradeoff")
        self.assertEqual(result["candidate_failures"],[])
        self.assertNotEqual(task(result)["development_decision"],"blocked")
        self.assertNotIn("primary_metric",task(result))
        self.assertEqual(result["objective_policy"]["status"],"provisional")

    def test_legacy_budgets_cannot_change_decision(self):
        report = fixture()
        report["datasets"][1]["metrics"] = {"accuracy":.79,"balanced_accuracy":.9}
        default = evaluate(report,bootstrap_iterations=100)
        explicit = evaluate(report,secondary_loss_budgets={"accuracy":0.0,"balanced_accuracy":1.0},classification_objective="accuracy",bootstrap_iterations=100)
        self.assertEqual(default["task_assessments"],explicit["task_assessments"])

    def test_severe_loss_not_hidden_by_average_or_other_task(self):
        report = fixture()
        report["datasets"][1]["metrics"]["balanced_accuracy"] = .5
        extra = deepcopy(report["datasets"])
        for row in extra:
            row["name"] = "gain"
        extra[1]["metrics"]["balanced_accuracy"] = .99
        report["datasets"] += extra
        reg = fixture("regression")["datasets"]
        reg[1]["metrics"]["rmse"] = 1
        report["datasets"] += reg
        result = assess(report,bootstrap_iterations=100)
        self.assertTrue(result["significant_regressions"])
        self.assertEqual(task(result,"regression")["outcome"],"improvement")
        self.assertEqual(task(result)["metrics"]["balanced_accuracy"]["worst_cases"][0]["name"],"case")

    def test_per_class_recall_alert(self):
        report = fixture()
        report["datasets"][0]["metrics"]["recall_by_class"] = {"minority":.8}
        report["datasets"][1]["metrics"]["recall_by_class"] = {"minority":.3}
        result=assess(report,bootstrap_iterations=100)
        self.assertTrue(any(a["metric"]=="recall_by_class" for a in result["significant_regressions"]))

    def test_equal_family_weights_not_seed_or_dataset_count(self):
        report=fixture()
        report["datasets"][1]["metrics"]["accuracy"] = .9
        for row in report["datasets"]:
            row["name"]="gain"
            row["family"]="repeated"
        many=[]
        for i in range(9):
            rows=deepcopy(report["datasets"])
            for row in rows: row["name"]=f"repeat{i}"
            many.extend(rows)
        loss=fixture()["datasets"]
        for row in loss: row["name"]="loss"; row["family"]="independent"
        loss[1]["metrics"]["accuracy"] = .6
        report["datasets"] = many+loss
        result=task(assess(report,bootstrap_iterations=100))
        self.assertEqual(result["family_count"],2)
        self.assertAlmostEqual(result["metrics"]["accuracy"]["family_weighted_improvement"],-.05)

    def test_single_family_has_no_confidence_claim(self):
        self.assertIsNone(family_interval([1.],100))
        result=task(assess(fixture(),bootstrap_iterations=100))
        self.assertIsNone(result["robust_ranking"]["interval95"])
        self.assertEqual(result["development_decision"],"retain_alternative")

    def test_zero_reference_loss_never_mixed_with_relative_reduction(self):
        report=fixture("regression")
        report["datasets"][0]["metrics"]["rmse"]=0
        report["datasets"][1]["metrics"]["rmse"]=1
        metric=task(assess(report,bootstrap_iterations=100),"regression")["metrics"]["rmse"]
        self.assertEqual(metric["zero_reference_cases"],1)
        self.assertIsNone(metric["family_weighted_improvement"])
        self.assertEqual(metric["worst_cases"][0]["raw_improvement"],-1)

    def test_missing_alternative_scenarios_explicit(self):
        result=task(assess(fixture("regression"),bootstrap_iterations=100),"regression")
        self.assertEqual(result["missing_scenarios"],["mae"])
        self.assertNotEqual(result["development_decision"],"promote")

    def test_missing_entire_family_detected_by_manifest(self):
        expected=[{"task":"classification","name":"case","seed":42},{"task":"classification","name":"omitted","seed":42}]
        result=assess(fixture(),manifest={"expected_cases":expected},bootstrap_iterations=100)
        self.assertEqual(result["reliability"]["decision"],"insufficient_evidence")

    def test_missing_pool_member_blocks_own_comparison(self):
        report=fixture()
        report["config"]["strategies"].append("routed")
        result=assess(report,manifest={"comparison_pool":["baseline","auto","routed"]},bootstrap_iterations=100)
        self.assertEqual(task(result)["outcome"],"inconclusive")

    def test_rank_pool_ties_and_directions(self):
        report=fixture("regression")
        report["datasets"][1]["metrics"]["rmse"]=1
        result=task(assess(report,bootstrap_iterations=100),"regression")["robust_ranking"]
        self.assertEqual(result["worst_regret"],{"baseline":1.,"auto":0.})
        self.assertEqual(result["improvement"],1)

    def test_completed_recovery_is_not_invalid_output(self):
        report=fixture()
        report["datasets"][1].update(fallback=True,fallback_kind="completed_baseline")
        result=assess(report,bootstrap_iterations=100)
        self.assertEqual(result["candidate_failures"],[])
        self.assertEqual(result["reliability"]["decision"],"review_required")
        self.assertEqual(result["cases"][0]["candidate_recovery"],"completed_baseline")

    def test_external_sixty_second_limit_even_if_harness_uses_120(self):
        report=fixture()
        report["config"]["timeout_seconds"]=120
        report["datasets"][1]["wall_seconds"]=65
        self.assertEqual(assess(report,bootstrap_iterations=100)["reliability"]["decision"],"blocked")

    def test_invalid_classification_does_not_erase_regression(self):
        report=fixture()
        report["datasets"][1]["valid"]=False
        reg=fixture("regression")["datasets"]
        reg[1]["metrics"]["rmse"]=1
        report["datasets"]+=reg
        result=assess(report,bootstrap_iterations=100)
        self.assertEqual(task(result)["development_decision"],"blocked")
        self.assertEqual(task(result,"regression")["outcome"],"improvement")

    def test_registry_mismatch_is_not_claimed_incumbent_comparison(self):
        result=task(assess(fixture(),registry={"tasks":{"classification":{"incumbent":{"strategy":"routed","source_sha256":"b"*64}}}},bootstrap_iterations=100))
        self.assertEqual(result["incumbent_comparison"],"not_established")

    def test_official_objective_not_replaced_by_proxy(self):
        policy={"status":"official","source":"published rules","source_sha256":"c"*64,"task_metrics":{"classification":{"official_quality_points":True}}}
        result=task(assess(fixture(),policy=policy,bootstrap_iterations=100))
        self.assertEqual(result["missing_scenarios"],["official_quality_points"])
        self.assertEqual(result["outcome"],"inconclusive")

    def test_future_official_promotion_requires_all_evidence(self):
        report=many_families()
        policy={"status":"official","source":"published rules","source_sha256":"c"*64,"task_metrics":{"classification":{"accuracy":True}}}
        registry={"tasks":{"classification":{"incumbent":{"strategy":"baseline","source_sha256":"b"*64}}}}
        manifest={"stage":"confirmation","predeclared":True,"previously_unopened":True,"frozen_source_sha256":"b"*64,
            "objective_policy_sha256":hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "comparison_pool":["baseline","auto"],"expected_cases":[{k:r[k] for k in ("task","name","seed")} for r in report["datasets"] if r["strategy"]=="baseline"],
            "verified_checks":{k:True for k in ("official_checker","profile_schema","maximum_size","memory_limit","clean_environment","immutable_baseline_compared","fallback_quality")}}
        good=assess(report,policy=policy,registry=registry,manifest=manifest,bootstrap_iterations=100)
        self.assertEqual(task(good)["development_decision"],"promote")
        manifest["previously_unopened"]=False
        self.assertNotEqual(task(assess(report,policy=policy,registry=registry,manifest=manifest,bootstrap_iterations=100))["development_decision"],"promote")

    def test_eight_families_supported_but_provisional_is_not_official(self):
        result=task(assess(many_families(),bootstrap_iterations=100))
        self.assertEqual(result["development_decision"],"advance_provisionally")
        self.assertGreater(result["robust_ranking"]["interval95"][0],0)

    def test_partial_source_and_bad_partition_fail_closed(self):
        report=fixture()
        report["datasets"][1]["partition"]=None
        self.assertEqual(task(assess(report,bootstrap_iterations=100))["outcome"],"inconclusive")
        report=fixture()
        del report["config"]["source_sha256_after"]
        self.assertEqual(assess(report,bootstrap_iterations=100)["reliability"]["decision"],"insufficient_evidence")

    def test_nonfinite_saved_metric_not_silently_ignored(self):
        report=fixture()
        report["datasets"][1]["metrics"]["accuracy"]=float("nan")
        result=assess(report,bootstrap_iterations=100)
        self.assertEqual(result["reliability"]["decision"],"insufficient_evidence")
        self.assertEqual(task(result)["outcome"],"inconclusive")

    def test_bad_pool_and_bad_identity_fail_closed(self):
        with self.assertRaises(ValueError): assess(fixture(),manifest={"comparison_pool":[{}]},bootstrap_iterations=100)
        report=fixture()
        report["datasets"][1]["task"]={"malformed":True}
        self.assertEqual(assess(report,bootstrap_iterations=100)["reliability"]["decision"],"insufficient_evidence")

    def test_quality_loss_with_speed_gain_retained_for_tradeoff(self):
        report=many_families()
        for row in report["datasets"]:
            if row["strategy"]=="auto":
                row["metrics"]={"accuracy":.79,"balanced_accuracy":.69,"macro_f1":.69}
                row["wall_seconds"]=3
        result=task(assess(report,bootstrap_iterations=100))
        self.assertEqual(result["outcome"],"harmful")
        self.assertEqual(result["development_decision"],"retain_alternative")
        self.assertLess(result["mean_extra_wall_seconds"],0)


class FrozenReferenceTests(unittest.TestCase):
    def test_same_strategy_across_versions_with_alias(self):
        reference=fixture()
        candidate=fixture()
        candidate["config"].update(source_sha256="c"*64,source_sha256_after="c"*64)
        original=deepcopy(candidate)
        paired=pair_reference(candidate,reference,"d"*64,"incumbent","auto","auto")
        self.assertEqual(candidate,original)
        registry={"tasks":{"classification":{"incumbent":{"strategy":"auto","source_sha256":"b"*64}}}}
        result=assess(paired,baseline="incumbent",candidate="auto",registry=registry,bootstrap_iterations=100)
        self.assertEqual(task(result)["incumbent_comparison"],"verified")
        self.assertEqual(result["reliability"]["decision"],"passed")

    def test_different_rows_or_unfinished_reference_rejected(self):
        reference=fixture()
        reference["datasets"][1]["data_sha256"]="e"*64
        with self.assertRaises(ValueError): pair_reference(fixture(),reference,"d"*64,"incumbent","auto","auto")
        reference=fixture()
        del reference["config"]["source_unchanged"]
        with self.assertRaises(ValueError): pair_reference(fixture(),reference,"d"*64,"incumbent","auto","auto")

    def test_cli_compares_registered_reference_version(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths={name:Path(directory)/(name+".json") for name in ("candidate","reference","registry","output")}
            candidate=fixture(); candidate["config"].update(source_sha256="c"*64,source_sha256_after="c"*64)
            paths["candidate"].write_text(json.dumps(candidate),encoding="utf-8")
            paths["reference"].write_text(json.dumps(fixture()),encoding="utf-8")
            paths["registry"].write_text(json.dumps({"tasks":{"classification":{"incumbent":{"strategy":"auto","source_sha256":"b"*64}}}}),encoding="utf-8")
            argv=["analyze.py",str(paths["candidate"]),"--reference-report",str(paths["reference"]),"--reference-run-strategy","auto",
                "--baseline-strategy","incumbent","--candidate-strategy","auto","--registry",str(paths["registry"]),"--output",str(paths["output"]),"--bootstrap-iterations","100"]
            with patch("sys.argv",argv),contextlib.redirect_stdout(io.StringIO()): self.assertEqual(analyze.main(),0)
            result=json.loads(paths["output"].read_text())["gate_assessment"]
            self.assertEqual(task(result)["incumbent_comparison"],"verified")


class PredictionEvidenceTests(unittest.TestCase):
    def test_confusion_macro_f1_and_opaque_labels(self):
        result=metrics("classification",["01","01","1","1"],["01","1","1","1"])
        self.assertEqual(result["accuracy"],.75)
        self.assertEqual(result["recall_by_class"],{"01":.5,"1":1.})
        self.assertAlmostEqual(result["macro_f1"],(2/3+.8)/2)

    def test_loss_metrics_and_adjusted_rand_label_invariance(self):
        self.assertEqual(metrics("regression",[0,2],[0,0]),{"rmse":2**.5,"mae":1.})
        self.assertEqual(metrics("clustering",[0,0,1,1],[1,1,0,0]),{"adjusted_rand_index":1.})

    def test_nonfinite_empty_and_misaligned_rejected(self):
        for truth,prediction in (([],[]),([0],[0,1]),([0],[float("nan")])):
            with self.assertRaises(ValueError): metrics("regression",truth,prediction)

    def test_evidence_hash_binding_and_no_mutation(self):
        report=fixture()
        for row in report["datasets"]:
            row["metrics"]={"accuracy":1.,"balanced_accuracy":1.}
        entry=dict(report["datasets"][0],truth=["01","1"],prediction=["01","1"])
        result=enrich(report,{"report_sha256":"d"*64,"cases":[entry]},"d"*64)
        self.assertIn("macro_f1",result["datasets"][0]["metrics"])
        self.assertNotIn("macro_f1",report["datasets"][0]["metrics"])
        with self.assertRaises(ValueError): enrich(report,{"report_sha256":"e"*64,"cases":[entry]},"d"*64)

    def test_existing_metric_disagreement_rejected(self):
        report=fixture()
        entry=dict(report["datasets"][0],truth=["x"],prediction=["x"])
        with self.assertRaises(ValueError): enrich(report,{"report_sha256":"d"*64,"cases":[entry]},"d"*64)

    def test_cli_preserves_existing_and_input_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source,output=Path(directory)/"report.json",Path(directory)/"score.json"
            source.write_text(json.dumps(fixture()),encoding="utf-8")
            output.write_text("historical",encoding="utf-8")
            for destination,extra in ((output,[]),(source,["--overwrite"])):
                with patch("sys.argv",["analyze.py",str(source),"--output",str(destination),*extra]),contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit): analyze.main()
            self.assertEqual(output.read_text(),"historical")

    def test_cli_evidence_enforcement_does_not_enforce_accuracy(self):
        report=fixture()
        report["datasets"][1]["metrics"]={"accuracy":.79,"balanced_accuracy":.9}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source,output=Path(directory)/"report.json",Path(directory)/"score.json"
            source.write_text(json.dumps(report),encoding="utf-8")
            argv=["analyze.py",str(source),"--output",str(output),"--enforce-evidence","--bootstrap-iterations","100"]
            with patch("sys.argv",argv),contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(analyze.main(),0)
            self.assertNotIn("explicit_strict_accuracy_diagnostic",json.loads(printed.getvalue()))
            self.assertEqual(json.loads(output.read_text())["gate_assessment"]["task_assessments"][0]["outcome"],"tradeoff")

    def test_cli_external_limit_failure_returns_nonzero(self):
        report=fixture()
        report["config"]["timeout_seconds"]=120
        report["datasets"][1]["wall_seconds"]=65
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source,output=Path(directory)/"report.json",Path(directory)/"score.json"
            source.write_text(json.dumps(report),encoding="utf-8")
            with patch("sys.argv",["analyze.py",str(source),"--output",str(output),"--enforce-evidence","--bootstrap-iterations","100"]),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(),2)


if __name__ == "__main__":
    unittest.main()
