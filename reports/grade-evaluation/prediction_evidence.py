"""Optional evaluator-owned predictions; no training and no production imports."""
from collections import Counter
from copy import deepcopy
import math


def numeric(values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def metrics(task, truth, prediction):
    if not isinstance(truth, list) or not isinstance(prediction, list) or not truth or len(truth) != len(prediction):
        raise ValueError("Aligned nonempty truth/prediction arrays are required")
    n = len(truth)
    if task == "classification":
        if not all(isinstance(v, str) for v in truth + prediction):
            raise ValueError("Classification evidence must preserve opaque string labels")
        labels = sorted(set(truth + prediction))
        confusion = {a: {b: 0 for b in labels} for a in labels}
        for a, b in zip(truth, prediction):
            confusion[a][b] += 1
        supports = Counter(truth)
        predicted = Counter(prediction)
        recalls = {a: confusion[a][a] / supports[a] for a in labels if supports[a]}
        f1 = [2 * confusion[a][a] / (supports[a] + predicted[a]) for a in labels if supports[a] + predicted[a]]
        return dict(accuracy=sum(a == b for a, b in zip(truth, prediction))/n,
                    balanced_accuracy=sum(recalls.values())/len(recalls), macro_f1=sum(f1)/len(f1),
                    recall_by_class=recalls, class_support=dict(supports), confusion_matrix=confusion)
    if not numeric(truth + prediction):
        raise ValueError("Numeric evidence must be finite")
    if task == "regression":
        mse = sum((a-b)**2 for a,b in zip(truth,prediction))/n
        return dict(rmse=math.sqrt(mse), mae=sum(abs(a-b) for a,b in zip(truth,prediction))/n)
    if task == "clustering":
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in truth + prediction):
            raise ValueError("Clustering evidence requires integer group labels")
        choose2 = lambda count: count*(count-1)/2
        table = Counter(zip(truth,prediction))
        a, b = sum(map(choose2,Counter(truth).values())), sum(map(choose2,Counter(prediction).values()))
        both = sum(map(choose2,table.values()))
        expected = a*b/choose2(n) if n > 1 else 0
        denominator = (a+b)/2-expected
        return {"adjusted_rand_index": (both-expected)/denominator if denominator else 1.0}
    raise ValueError("Unsupported task")


def enrich(report, evidence, report_sha256):
    """Attach independent diagnostics only to a hash-matched report copy."""
    if not isinstance(evidence, dict) or evidence.get("report_sha256") != report_sha256:
        raise ValueError("Prediction evidence is not bound to this report hash")
    result = deepcopy(report)
    rows = result.get("datasets", [])
    index = {}
    for row in rows:
        key = (row["task"],row["name"],row["seed"],row["strategy"])
        if key in index:
            raise ValueError("Report contains duplicate prediction-evidence identity")
        index[key] = row
    entries = evidence.get("cases")
    if not isinstance(entries,list):
        raise ValueError("Prediction evidence cases must be a list")
    seen = set()
    for entry in entries:
        key = (entry["task"],entry["name"],entry["seed"],entry["strategy"])
        if key in seen or key not in index:
            raise ValueError("Duplicate or unmatched prediction evidence")
        seen.add(key)
        row = index[key]
        if row.get("data_sha256") != entry.get("data_sha256"):
            raise ValueError("Prediction evidence input hash mismatch")
        calculated = metrics(entry["task"],entry["truth"],entry["prediction"])
        original = row.get("metrics", {})
        for name,value in calculated.items():
            if isinstance(value, (int,float)) and name in original and not math.isclose(original[name],value,abs_tol=1e-9,rel_tol=1e-9):
                raise ValueError(f"Saved prediction metrics disagree: {name}")
        row["metrics"] = {**original,**calculated}
    return result
