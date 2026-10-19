"""Frozen clustering confirmation; reference labels remain in this harness."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from datasets import make_broad_cases
from evaluate import evaluate_case


def fingerprints():
    paths = [ROOT / "src/unknown_dataset/clustering.py",
             ROOT / "tests/test_clustering_candidates.py", Path(__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def main():
    before = fingerprints()
    rows = []
    for seed in (2027, 4099):
        work = ROOT / "experiments/clustering-improvement-confirmation-input" / str(seed)
        cases = [c for c in make_broad_cases(work, seed) if c["task"] == "clustering"]
        # Same ellipse family as development, with independently generated rows.
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame(np.vstack([
            rng.multivariate_normal(center, [[2, 1.95], [1.95, 2]], 200)
            for center in [[-2, 2], [0, 0], [2, -2]]]), columns=["x", "y"])
        folder = work / "ellipses"
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / "train.csv", index=False)
        (folder / "metadata.json").write_text(json.dumps({"task": "clustering",
            "n_clusters": 3, "feature_types": {"x": "numeric", "y": "numeric"}}))
        cases.append(dict(name="ellipses", family="ellipses", task="clustering",
            input=folder, reference=np.repeat(np.arange(3), 200), n_clusters=3,
            source="synthetic", partition="fresh confirmation", seed=seed))
        for case in cases:
            for strategy in ("baseline", "auto"):
                output = work / f"output-{case['name']}-{strategy}"
                row = evaluate_case(case, output, seed, timeout=60, budget=50, strategy=strategy)
                rows.append(row)
                print(seed, case["name"], strategy, row["valid"], row["metrics"], flush=True)
    report = {"notice": "Seeds 2027/4099 predeclared before evaluation; no post-confirmation tuning.",
              "fingerprints_before": before, "fingerprints_after": fingerprints(), "datasets": rows}
    report["owned_source_unchanged"] = report["fingerprints_before"] == report["fingerprints_after"]
    (ROOT / "experiments/clustering-improvement-confirmation.json").write_text(json.dumps(report, indent=2))
    return 0 if all(r["valid"] for r in rows) and report["owned_source_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
