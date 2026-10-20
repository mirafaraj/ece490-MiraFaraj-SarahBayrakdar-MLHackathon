from copy import deepcopy
import numpy as np
from scripts.second_classification import choose, digest, raw_metrics, record


def test_independent_metrics_lexical_labels_and_missing_class():
    truth = np.array(["001", "001", "1", "1"])
    prediction = np.array(["001", "1", "1", "other"])
    result = record(truth, prediction, ["001", "1", "other"])
    assert result["independent_metrics"]["accuracy"] == .5
    assert result["metrics"]["class_recall"] == [.5, .5, 0]


def test_shared_cap_excludes_incomplete_second_fold():
    records = {name: [{"seconds": 2., "metrics": {"accuracy": value}}] * 2
               for name, value in [("extra_trees", .5), ("linear", .8), ("histogram", .9)]}
    one = choose(records, "accuracy", 1, cap=5)
    two = choose(records, "accuracy", 2, cap=5)
    assert one["selected"] == "histogram"
    assert two["selected"] == "extra_trees"
    assert two["excluded_candidates"] == {"linear": "shared selection cap exhausted",
                                           "histogram": "shared selection cap exhausted"}


def test_outer_fields_and_evaluation_labels_cannot_change_choice():
    rows = {name: [{"seconds": .1, "metrics": {"accuracy": value}, "outer": .1}]
            for name, value in [("extra_trees", .6), ("linear", .7), ("histogram", .5)]}
    changed = deepcopy(rows)
    for folds in changed.values():
        folds[0]["outer"] = {"truth": ["changed"], "accuracy": 1}
    assert choose(rows, "accuracy", 1) == choose(changed, "accuracy", 1)


def test_input_hash_stable_and_value_sensitive():
    assert digest({"a": [1, 2], "b": None}) == digest({"b": None, "a": [1, 2]})
    assert digest({"a": [1, 2]}) != digest({"a": [2, 1]})
