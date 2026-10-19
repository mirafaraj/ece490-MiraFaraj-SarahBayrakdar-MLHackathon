"""Offline diagnostics must report errors and empirical regret without an oracle selector."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.selector_stability import diagnostics


def row(name, score, valid=True, candidate=None):
    return {"name": "case", "split_seed": 42, "task": "regression", "strategy": name,
            "score": score, "valid": valid, "selected_candidate": candidate,
            "validation_scores": {candidate: 1.5} if candidate else {}, "fallback_kind": "none",
            "wall_seconds": 1.0, "budget_exceeded": False, "error": None if valid else "failed"}


def test_gap_includes_dummy_and_validation_loss_direction():
    rows = [row("dummy", -1), row("extra_trees", -3), row("linear", -2),
            row("histogram", -4), row("baseline", -3), row("routed", -2, candidate="linear")]
    result = diagnostics(rows)
    diagnostic = result["per_split"][0]
    assert diagnostic["best_evaluated_gap"] == 1
    assert diagnostic["comparison_complete"]
    assert diagnostic["validation_quality"] == -1.5
    assert diagnostic["validation_minus_outer_quality"] == .5
    assert result["selection_frequencies"]["routed"] == {"linear": 1}


def test_missing_candidate_comparison_marked_incomplete():
    rows = [row("dummy", -1), row("extra_trees", -3), row("linear", None, False),
            row("histogram", -4), row("baseline", -3), row("auto", -3, candidate="extra_trees")]
    diagnostic = diagnostics(rows)["per_split"][0]
    assert not diagnostic["comparison_complete"]
    assert diagnostic["missing_or_failed_candidates"] == ["linear"]
    assert diagnostic["outcome"] == "tie"

