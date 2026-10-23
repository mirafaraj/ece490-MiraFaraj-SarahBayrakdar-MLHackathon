"""Fixed diagnostic comparison. No downloads; reference arrays remain local."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
POLICIES = {"full": {}, "quarter": {"clustering_indicator_amplitude": .25},
            "omitted": {"clustering_missing_indicators": False}}
SEEDS = [104729, 104743, 104759]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def prepare():
    import numpy as np
    import pandas as pd
    from second_clustering import historical_losses, load_historical_case
    records = []

    def add(identity, family, condition, seed, frame, reference, k, provenance):
        folder = HERE / "local" / "inputs" / identity
        folder.mkdir(parents=True, exist_ok=False)
        frame.to_csv(folder / "train.csv", index=False, float_format="%.17g")
        types = {str(c): "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "categorical" for c in frame}
        save(folder / "metadata.json", dict(task="clustering", n_clusters=k, feature_types=types))
        if reference is not None:
            save(HERE / "local" / "references" / (identity + ".json"), np.asarray(reference).tolist())
        records.append(dict(id=identity, family=family, condition=condition, seed=seed, k=k,
            rows=len(frame), features=len(frame.columns), reference=reference is not None,
            source=provenance, input_sha256={p.name: sha(p) for p in folder.iterdir()}))

    old_path = ROOT / "experiments/third-batch/clustering.json"
    old = json.loads(old_path.read_text())
    for case in old["cases"]:
        if ((case["family"] in ("mixed", "anisotropic_noise") and case["masking"] in ("random", "informative"))
                or (case["family"] in ("correlated", "curved") and case["masking"] == "none" and case["seed"] == 99013)):
            frame = pd.DataFrame(case["input"]["data"], columns=case["input"]["columns"])
            add(f"old-{case['family']}-{case['masking']}-{case['seed']}", case["family"], case["masking"],
                case["seed"], frame, case["reference"], case["clusters"], dict(path=str(old_path), sha256=sha(old_path)))
    report = json.loads((ROOT / "experiments/first-batch/clustering.json").read_text())
    historical = next(c for c, _ in historical_losses(report)
                      if c["family"] == "missing_axis" and c["seed"] == 6101 and c["mechanism"] == "none")
    frame, reference, k = load_historical_case(historical)
    add("old-missing-axis-6101", "missing_axis", "intrinsic_random", 6101, frame, reference, k,
        dict(path="experiments/fixtures/historical-clustering-inputs-v1.npz", historical=True))
    for family in ["mask_only", "nuisance_blocks"]:
        for imbalance in [False, True]:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                counts = [540, 54, 6] if imbalance else [200, 200, 200]
                reference = np.repeat(np.arange(3), counts)
                if family == "mask_only":
                    mask = np.repeat(np.eye(3, dtype=bool)[reference], 4, axis=1)
                    mask ^= rng.random(mask.shape) < .05
                    values = np.full((600, 12), 7.)
                    values[mask] = np.nan
                    frame = pd.DataFrame(values, columns=[f"mask{j}" for j in range(12)])
                else:
                    centers = np.array([[-4., 0.], [0., 4.], [4., 0.]])
                    frame = pd.DataFrame(centers[reference] + rng.normal(size=(600, 2)), columns=["x", "y"])
                    sites = rng.integers(0, 3, 600)  # Independent of intended groups.
                    noise = rng.normal(size=(600, 8))
                    probabilities = np.column_stack([np.where(sites == j % 3, .8, .05) for j in range(8)])
                    noise[rng.random(noise.shape) < probabilities] = np.nan
                    for j in range(8):
                        frame[f"nuisance{j}"] = noise[:, j]
                condition = "rare" if imbalance else "balanced"
                add(f"{family}-{condition}-{seed}", family, condition, seed, frame, reference, 3,
                    dict(generator="compare.py::prepare", counts=counts, labels_used_for_mask=family == "mask_only"))
    mall_path = Path(r"C:\Users\User\Downloads\Mall_Customers.csv")
    mall = pd.read_csv(mall_path).drop(columns="CustomerID")
    for condition in ["clean", "random30"]:
        frame = mall.copy()
        if condition == "random30":
            cols = frame.select_dtypes(include="number").columns
            frame = frame.astype({c: float for c in cols})
            frame.loc[:, cols] = frame[cols].mask(np.random.default_rng(SEEDS[0]).random((len(frame), len(cols))) < .3)
        add("mall-" + condition, "Mall", condition, SEEDS[0], frame, None, 5,
            dict(path=str(mall_path), sha256=sha(mall_path), k_assumption=5, excluded="CustomerID"))
    protocol = dict(registered_before_fitting=True, policies=POLICIES, cases=records,
        existing_cases=11, new_conditions=12, mall_conditions=2, main_calls=len(records) * 3,
        mall_subsample_calls=30, mall_subsample_fraction=.8, mall_subsample_seeds=[41, 43, 47, 53, 59],
        budget=20, cli_budget=50, external_limit=60, fresh_confirmation_opened=False,
        seeds=SEEDS, estimator_changes=False, gates="STAGE_1_PROPOSAL.md::Fixed proposed gates",
        preparation_scope="Approved minimal comparison; no downloads or masked large-input expansion",
        packages={n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn"]},
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            [Path(__file__), ROOT / "src/unknown_dataset/clustering.py", ROOT / "src/unknown_dataset/preprocessing.py",
             ROOT / "src/unknown_dataset/models.py", ROOT / "src/unknown_dataset/runner.py"]})
    save(HERE / "protocol.json", protocol)
    print("Registered", len(records), "cases,", protocol["main_calls"], "main calls plus 30 Mall stability fits", flush=True)


def run():
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from unknown_dataset.clustering import fit_predict_clustering
    from unknown_dataset.contract import load_dataset
    from unknown_dataset.preprocessing import TableCleaner, build_preprocessor
    from unknown_dataset.profiling import build_profile
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert not (HERE / "local/results.json").exists()
    rows = []
    started = time.monotonic()
    for index, case in enumerate(protocol["cases"]):
        folder = HERE / "local/inputs" / case["id"]
        assert all(sha(folder / p) == h for p, h in case["input_sha256"].items())
        data = load_dataset(folder)
        profile = build_profile(data.X_train, data.feature_types)
        profile_before = json.dumps(profile, sort_keys=True, allow_nan=False)
        save(HERE / "local/profiles" / (case["id"] + ".json"), profile)
        reference = np.array(json.loads((HERE / "local/references" / (case["id"] + ".json")).read_text())) if case["reference"] else None
        common = None
        if reference is None:
            cleaner = TableCleaner(data.feature_types).fit(data.X_train)
            processor = build_preprocessor(cleaner, scale_numeric=True)
            common = processor.fit_transform(cleaner.transform(data.X_train))
            common = common.toarray() if sparse.issparse(common) else common
        names = list(POLICIES)
        names = names[index % 3:] + names[:index % 3]
        for policy in names:
            start = time.monotonic()
            pred, info = fit_predict_clustering("clustering", data.X_train, feature_types=data.feature_types,
                n_clusters=case["k"], seed=case["seed"], time_budget=20,
                experimental_controls={**POLICIES[policy], "runtime_instrumentation": True})
            elapsed = time.monotonic() - start
            valid = len(pred) == case["rows"] and set(pred) == set(range(case["k"]))
            row = dict(id=case["id"], family=case["family"], condition=case["condition"], seed=case["seed"],
                policy=policy, valid=valid, direct_seconds=elapsed, group_sizes=np.bincount(pred, minlength=case["k"]).tolist(),
                repairs=info["empty_clusters_repaired"], fallback=bool(info.get("fallback", False)), info=info)
            save(HERE / "local/predictions" / case["id"] / (policy + ".json"), pred.tolist())
            if reference is not None:
                row.update(ari=adjusted_rand_score(reference, pred), ami=adjusted_mutual_info_score(reference, pred))
                table = np.zeros((case["k"], case["k"]), dtype=int)
                np.add.at(table, (reference, pred), 1)
                ii, jj = linear_sum_assignment(-table)
                recalls = table[ii, jj] / table.sum(axis=1)[ii]
                row.update(matched_group_recall=recalls.tolist(), macro_group_recall=float(recalls.mean()),
                    smallest_group_recall=float(recalls[np.argmin(table.sum(axis=1)[ii])]))
            else:
                row.update(silhouette_common=silhouette_score(common, pred),
                    calinski_harabasz_common=calinski_harabasz_score(common, pred),
                    davies_bouldin_common=davies_bouldin_score(common, pred))
                stability = []
                for sub_seed in protocol["mall_subsample_seeds"]:
                    ix = np.sort(np.random.default_rng(sub_seed).choice(len(pred), int(.8 * len(pred)), replace=False))
                    sub_pred, sub_info = fit_predict_clustering("clustering", data.X_train.iloc[ix],
                        feature_types=data.feature_types, n_clusters=5, seed=case["seed"], time_budget=20,
                        experimental_controls=POLICIES[policy])
                    # ARI compares two fitted partitions on the same rows: stability, NOT ground-truth recovery.
                    stability.append(dict(seed=sub_seed, partition_stability_ari=adjusted_rand_score(pred[ix], sub_pred),
                        group_sizes=np.bincount(sub_pred, minlength=5).tolist(), info=sub_info))
                    save(HERE / "local/stability" / case["id"] / policy / (str(sub_seed) + ".json"), sub_pred.tolist())
                row["stability"] = stability
                row["mean_partition_stability"] = float(np.mean([s["partition_stability_ari"] for s in stability]))
            assert profile_before == json.dumps(build_profile(data.X_train, data.feature_types), sort_keys=True, allow_nan=False)
            assert valid and not row["fallback"]
            rows.append(row)
            with (HERE / "local/attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(case["id"], "completed", flush=True)
    save(HERE / "local/results.json", dict(rows=rows, elapsed_seconds=time.monotonic() - started,
        timing_note="Per-case direct times exclude process startup. CLI checks are separate.",
        memory_note="No per-case peak memory measurement available; process-level counters are separate."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "run"])
    phase = parser.parse_args().phase
    if phase == "prepare":
        prepare()
    else:
        run()
