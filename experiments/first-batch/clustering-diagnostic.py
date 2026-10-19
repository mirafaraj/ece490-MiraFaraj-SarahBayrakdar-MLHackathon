"""Repeat subset and reproduce inspected original missing-axis rows, no file generation."""
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import batch_clustering as batch
import datasets

output = Path(__file__).with_name("clustering-diagnostic.json")
if output.exists():
    raise FileExistsError(output)
original = json.loads(output.with_name("clustering.json").read_text())
rows = []
for mechanism, fraction in batch.CONDITIONS:
    x, labels, k = batch.generate("missing_axis", 6101, mechanism, fraction)
    for policy, controls in batch.POLICIES.items():
        pred, info = batch.fit_predict_clustering("clustering", x, n_clusters=k,
            seed=6101, time_budget=20, experimental_controls=controls)
        before = next(row for row in original["rows"] if row["family"] == "missing_axis"
            and row["seed"] == 6101 and row["mechanism"] == mechanism
            and row["fraction"] == fraction and row["policy"] == policy)
        digest = hashlib.sha256(np.asarray(pred, dtype="int64").tobytes()).hexdigest()
        rows.append(dict(kind="repeat", mechanism=mechanism, fraction=fraction,
            policy=policy, identical=digest == before["prediction_sha256"]))
# Intercept only the generator's writing stage, leaving its exact RNG consumption
# and case construction intact; model receives features and required k only.
datasets._write_cases = lambda root, seed, definitions: definitions
for seed in (42, 8081):
    cases = datasets.make_router_cases(Path("unused"), seed)
    _, _, x, reference, _, k = next(c for c in cases if c[0] == "missing_cluster_axes")
    x = x.copy()
    x.columns = [str(c) for c in x.columns]  # Match original _write_cases normalization.
    for policy, controls in batch.POLICIES.items():
        start = time.monotonic()
        pred, info = batch.fit_predict_clustering("clustering", x, n_clusters=k,
            seed=seed, time_budget=50, experimental_controls=controls)
        rows.append(dict(kind="inspected_original_diagnostic", seed=seed, policy=policy,
            ari=adjusted_rand_score(reference, pred), ami=adjusted_mutual_info_score(reference, pred),
            valid=len(np.unique(pred)) == k, wall_seconds=time.monotonic()-start, info=info))
sources = [Path(__file__), Path(datasets.__file__)]
output.write_text(json.dumps(dict(rows=rows, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sources}), indent=2))
print("Repeat matches", sum(r.get("identical", False) for r in rows), "/20")
for row in rows:
    if "ari" in row:
        print(row["seed"], row["policy"], round(row["ari"], 5), row["info"]["model"])
