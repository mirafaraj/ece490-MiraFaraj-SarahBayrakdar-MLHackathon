"""Run with `python -m unittest discover -s reports/grade-evaluation -p test_promotion.py`."""
from copy import deepcopy
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import analyze
from promotion import strict_evaluate as evaluate
from promotion import evaluate as balanced_evaluate


def fixture(task="classification", candidate="auto"):
    metrics = {"classification": {"accuracy": .8, "balanced_accuracy": .7},
               "regression": {"rmse": 2.0}, "clustering": {"adjusted_rand_index": .5}}[task]
    rows = [dict(seed=42, task=task, name="case", strategy=s, valid=True, metrics=dict(metrics),
                 data_sha256="a"*64, partition="development", fallback=False, exit_code=0,
                 wall_seconds=5., error=None) for s in ("baseline", candidate)]
    return dict(config=dict(source_sha256="b"*64, source_sha256_after="b"*64, source_unchanged=True,
                            timeout_seconds=60, seeds=[42], strategies=["baseline", candidate]), datasets=rows)


class GateTests(unittest.TestCase):
    def decision(self, report, **kwargs):
        return evaluate(report, **kwargs)["measured_accuracy_gate"]["decision"]

    def test_case_loss_not_masked_by_mean(self):
        report = fixture()
        report["datasets"][1]["metrics"]["accuracy"] -= .01
        extra = deepcopy(report["datasets"])
        for row in extra:
            row["name"] = "big_gain"
        extra[1]["metrics"]["accuracy"] = .99
        report["datasets"] += extra
        result = evaluate(report)
        self.assertGreater(result["measured_accuracy_gate"]["mean_accuracy_change"], 0)
        self.assertEqual(result["measured_accuracy_gate"]["decision"], "rejected")

    def test_tolerance(self):
        report = fixture()
        report["datasets"][1]["metrics"]["accuracy"] -= 5e-13
        self.assertEqual(self.decision(report), "passed")
        report["datasets"][1]["metrics"]["accuracy"] -= 2e-12
        self.assertEqual(self.decision(report), "rejected")

    def test_no_drop_is_not_promotion(self):
        result = evaluate(fixture())
        self.assertEqual(result["measured_accuracy_gate"]["decision"], "passed")
        self.assertEqual(result["promotion"]["decision"], "insufficient_evidence")
        self.assertFalse(result["promotion"]["hidden_set_guarantee"])

    def test_invalid_rows(self):
        report = fixture()
        report["datasets"][1]["valid"] = False
        self.assertEqual(self.decision(report), "rejected")

    def test_missing_nonfinite_metrics(self):
        for bad in (None, float("nan"), float("inf"), True):
            with self.subTest(bad=bad):
                report = fixture()
                report["datasets"][1]["metrics"]["accuracy"] = bad
                self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_missing_secondary_metric(self):
        report = fixture()
        del report["datasets"][1]["metrics"]["balanced_accuracy"]
        self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_arbitrary_strategy_and_extra_rows(self):
        report = fixture(candidate="routed")
        unused = deepcopy(report["datasets"][1])
        unused["strategy"] = "auto"
        unused["valid"] = False
        report["datasets"].append(unused)
        self.assertEqual(self.decision(report, candidate="routed"), "passed")

    def test_subset_tasks(self):
        for task in ("regression", "clustering"):
            with self.subTest(task=task):
                result = evaluate(fixture(task))
                self.assertEqual(result["measured_accuracy_gate"]["decision"], "insufficient_evidence")
                self.assertIn("classification", result["missing_coverage"]["tasks"])

    def test_bad_input_hash(self):
        report = fixture()
        report["datasets"][1]["data_sha256"] = "c"*64
        self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_partial_or_changed_source(self):
        for field in ("source_unchanged", "source_sha256_after"):
            report = fixture()
            del report["config"][field]
            self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_missing_pair_or_seed(self):
        report = fixture()
        report["datasets"].pop()
        self.assertEqual(self.decision(report), "insufficient_evidence")
        report = fixture()
        report["config"]["seeds"].append(314)
        self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_fallback_regression(self):
        report = fixture()
        report["datasets"][1]["fallback"] = True
        self.assertEqual(self.decision(report), "rejected")
        report["datasets"][0]["fallback"] = True
        self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_deadline_and_missing_time(self):
        report = fixture()
        report["datasets"][1]["wall_seconds"] = 60
        self.assertEqual(self.decision(report), "rejected")
        report["datasets"][1]["wall_seconds"] = None
        self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_duplicate_row(self):
        report = fixture()
        report["datasets"].append(deepcopy(report["datasets"][0]))
        self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_known_loss_with_unrelated_missing_metadata(self):
        report = fixture()
        report["datasets"][1]["metrics"]["accuracy"] = .7
        del report["datasets"][1]["metrics"]["balanced_accuracy"]
        report["datasets"][1]["wall_seconds"] = None
        result = evaluate(report)
        self.assertEqual(result["measured_accuracy_gate"]["decision"], "rejected")
        self.assertTrue(result["evidence_gaps"])

    def test_malformed_artifact(self):
        for report in (None, {"datasets": None}, {"config": None, "datasets": [None]}):
            self.assertEqual(self.decision(report), "insufficient_evidence")
        report = fixture()
        report["datasets"][1]["metrics"] = None
        self.assertEqual(self.decision(report), "insufficient_evidence")

    def test_baseline_cannot_be_candidate(self):
        with self.assertRaises(ValueError):
            evaluate(fixture(), candidate="baseline")

    def test_cli_default_reports_but_enforcement_fails(self):
        report = fixture()
        report["datasets"][1]["metrics"]["accuracy"] = .7
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source, output = Path(directory)/"report.json", Path(directory)/"score.json"
            source.write_text(json.dumps(report), encoding="utf-8")
            argv = ["analyze.py", str(source), "--output", str(output)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(), 0)
            with patch("sys.argv", argv+["--enforce-accuracy-gate","--overwrite"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(), 2)
            self.assertEqual(json.loads(output.read_text())["gate_assessment"]["measured_accuracy_gate"]["decision"], "rejected")

    def test_cli_malformed_json_nonzero_enforcement(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source, output = Path(directory)/"partial.json", Path(directory)/"score.json"
            source.write_text('{"datasets": [', encoding="utf-8")
            with patch("sys.argv", ["analyze.py", str(source), "--output", str(output), "--enforce-accuracy-gate"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(), 2)
            self.assertEqual(json.loads(output.read_text())["gate_assessment"]["measured_accuracy_gate"]["decision"], "insufficient_evidence")


class BalancedAssessmentTests(unittest.TestCase):
    def task(self, result, task="classification"):
        return next(row for row in result["task_assessments"] if row["task"] == task)

    def test_accuracy_loss_preserves_balanced_gain_as_tradeoff(self):
        report = fixture()
        report["datasets"][1]["metrics"] = {"accuracy": .79, "balanced_accuracy": .85}
        result = balanced_evaluate(report)
        self.assertEqual(result["measured_accuracy_gate"]["decision"], "rejected")
        self.assertEqual(result["candidate_failures"], [])
        self.assertEqual(self.task(result)["outcome"], "tradeoff")
        self.assertEqual(self.task(result)["development_decision"], "retain_alternative")
        self.assertEqual(result["promotion"]["decision"], "insufficient_evidence")

    def test_predeclared_secondary_budget(self):
        report = fixture()
        report["datasets"][1]["metrics"] = {"accuracy": .79, "balanced_accuracy": .85}
        result = balanced_evaluate(report, secondary_loss_budgets={"accuracy":.02})
        self.assertEqual(self.task(result)["development_decision"], "retain_alternative")
        self.assertFalse(result["legacy_diagnostics"]["affects_promotion"])
        self.assertEqual(self.task(result)["outcome"], "tradeoff")
        self.assertFalse(result["promotion"]["hidden_set_guarantee"])

    def test_both_metrics_worse_is_harmful(self):
        report = fixture()
        report["datasets"][1]["metrics"] = {"accuracy": .7, "balanced_accuracy": .6}
        self.assertEqual(self.task(balanced_evaluate(report))["outcome"], "harmful")

    def test_pareto_improvement(self):
        report = fixture()
        report["datasets"][1]["metrics"]["balanced_accuracy"] = .85
        self.assertEqual(self.task(balanced_evaluate(report))["outcome"], "improvement")

    def test_regression_is_preserved_independently(self):
        report = fixture()
        report["datasets"][1]["metrics"]["accuracy"] = .7
        regression = fixture("regression")
        regression["datasets"][1]["metrics"]["rmse"] = 1.0
        report["datasets"] += regression["datasets"]
        result = balanced_evaluate(report)
        self.assertEqual(self.task(result,"regression")["outcome"], "improvement")
        self.assertEqual(self.task(result,"regression")["development_decision"], "retain_alternative")

    def test_no_classification_does_not_block_regression(self):
        result = balanced_evaluate(fixture("regression"))
        self.assertEqual(self.task(result,"classification")["outcome"], "inconclusive")
        self.assertEqual(self.task(result,"regression")["development_decision"], "retain_alternative")
        self.assertEqual(result["reliability"]["decision"], "passed")

    def test_reliability_and_source_still_block(self):
        report = fixture()
        report["datasets"][1]["fallback"] = True
        self.assertEqual(balanced_evaluate(report)["reliability"]["decision"], "review_required")
        report = fixture()
        report["config"]["source_unchanged"] = False
        self.assertEqual(self.task(balanced_evaluate(report))["outcome"], "inconclusive")

    def test_reference_strategy_can_be_auto(self):
        report = fixture(candidate="routed")
        report["datasets"][0]["strategy"] = "auto"
        report["config"]["strategies"] = ["auto","routed"]
        result = balanced_evaluate(report,candidate="routed",baseline="auto")
        self.assertEqual(self.task(result)["outcome"], "equivalent")

    def test_invalid_loss_budgets(self):
        for budget in ({"accuracy":-1},{"accuracy":float("nan")},{"unknown":.1}):
            with self.assertRaises(ValueError):
                balanced_evaluate(fixture(), secondary_loss_budgets=budget)

    def test_equivalent_slower_has_explicit_cost_verdict(self):
        report = fixture()
        report["datasets"][1]["wall_seconds"] = 7
        result = self.task(balanced_evaluate(report))
        self.assertEqual(result["outcome"], "equivalent")
        self.assertEqual(result["observed_cost_verdict"], "dominated_on_measured_quality_and_time")

    def test_zero_budget_no_loss_retains_improvement(self):
        report = fixture()
        report["datasets"][1]["metrics"] = {"accuracy":.85,"balanced_accuracy":.8}
        result = self.task(balanced_evaluate(report))
        self.assertEqual(result["development_decision"], "retain_alternative")
        self.assertNotIn("secondary_loss_budget",result)

    def test_task_metadata_gap_blocks_own_task(self):
        report = fixture()
        regression = fixture("regression")
        regression["datasets"][1]["metrics"]["rmse"] = 1.0
        report["datasets"] += regression["datasets"]
        report["datasets"][1]["wall_seconds"] = None
        result = balanced_evaluate(report)
        self.assertEqual(self.task(result)["outcome"], "inconclusive")
        self.assertEqual(self.task(result,"regression")["outcome"], "improvement")

    def test_summary_mismatch_blocks_new_decisions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source, summary, output = [Path(directory)/name for name in ("report.json","summary.json","output.json")]
            source.write_text(json.dumps(fixture()),encoding="utf-8")
            summary.write_text(json.dumps({"source_sha256":"wrong"}),encoding="utf-8")
            argv=["analyze.py",str(source),"--summary",str(summary),"--output",str(output)]
            with patch("sys.argv",argv),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(),0)
            result=json.loads(output.read_text())["gate_assessment"]
            self.assertEqual(result["reliability"]["decision"],"insufficient_evidence")
            self.assertEqual(result["promotion"]["decision"],"insufficient_evidence")
            self.assertTrue(all(t["development_decision"]=="insufficient_evidence" for t in result["task_assessments"]))

    def test_cli_missing_metric_object_writes_insufficient_decision(self):
        report = fixture()
        report["datasets"][1]["metrics"] = None
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            source, output = Path(directory)/"report.json", Path(directory)/"score.json"
            source.write_text(json.dumps(report), encoding="utf-8")
            with patch("sys.argv", ["analyze.py", str(source), "--output", str(output), "--enforce-accuracy-gate"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analyze.main(), 2)
            self.assertEqual(json.loads(output.read_text())["gate_assessment"]["measured_accuracy_gate"]["decision"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
