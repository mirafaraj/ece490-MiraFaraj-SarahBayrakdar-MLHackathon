"""Offline benchmark tables. References are kept exclusively in the harness."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes, load_iris, make_classification, make_regression, make_friedman1, make_blobs, make_moons
from sklearn.model_selection import train_test_split


def make_broad_cases(root: Path, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    definitions = []
    for name, loader, task in (("breast_cancer", load_breast_cancer, "classification"),
                               ("wine", load_wine, "classification"),
                               ("diabetes", load_diabetes, "regression")):
        data = loader()
        definitions.append((name, task, pd.DataFrame(data.data, columns=data.feature_names), data.target,
                            "sklearn bundled real dataset", None))
    x, y = make_classification(n_samples=750, n_features=18, n_informative=6, weights=[.94, .06],
                               flip_y=.025, random_state=seed)
    definitions.append(("imbalanced", "classification", pd.DataFrame(x), y, "synthetic", None))
    category = rng.integers(0, 70, 800)
    x = pd.DataFrame({"value": rng.normal(size=800), "category": [f"category_{v}" for v in category]})
    y = np.where(x["value"].to_numpy() + (category % 3) > 1, .25, 1.75)
    x.loc[rng.random(800) < .12, "value"] = np.nan
    definitions.append(("high_cardinality", "classification", x, y, "synthetic", None))
    x, y = make_regression(n_samples=750, n_features=16, n_informative=10, noise=12, random_state=seed)
    definitions.append(("linear", "regression", pd.DataFrame(x), y, "synthetic", None))
    x, y = make_friedman1(n_samples=750, n_features=14, noise=1.0, random_state=seed)
    definitions.append(("nonlinear", "regression", pd.DataFrame(x), y, "synthetic", None))
    v = rng.normal(size=800)
    category = rng.choice(["basic", "premium", "standard"], 800)
    y = 4 * v ** 2 + 3 * (category == "premium") + rng.normal(0, .8, 800)
    x = pd.DataFrame({"value": v, "category": category, "noise": rng.normal(size=800), "constant": 1})
    x.loc[rng.random(800) < .12, "value"] = np.nan
    x["duplicate"] = x["value"]
    definitions.append(("mixed_missing", "regression", x, y, "synthetic", None))
    x, y = make_blobs(n_samples=660, centers=4, n_features=4, cluster_std=1.8, random_state=seed)
    definitions.append(("noisy_blobs", "clustering", pd.DataFrame(np.column_stack([x, rng.normal(0, 2, (660, 4))])), y, "synthetic", 4))
    x, y = make_blobs(n_samples=660, centers=3, cluster_std=[1.0, 1.8, .6], random_state=seed)
    x = x @ np.array([[.6, -.6], [-.4, .8]])
    definitions.append(("anisotropic", "clustering", pd.DataFrame(x), y, "synthetic", 3))
    x, y = make_moons(n_samples=660, noise=.07, random_state=seed)
    definitions.append(("moons", "clustering", pd.DataFrame(x), y, "synthetic", 2))
    x, y = make_blobs(n_samples=660, centers=3, cluster_std=1.5, random_state=seed)
    frame = pd.DataFrame({"x": x[:, 0], "scaled": x[:, 1] * 1000,
                          "category": np.array(["a", "b", "c"])[y], "noise": rng.normal(size=660)})
    flip = rng.random(660) < .25
    frame.loc[flip, "category"] = rng.choice(["a", "b", "c"], flip.sum())
    frame.loc[rng.random(660) < .1, "x"] = np.nan
    definitions.append(("mixed_clusters", "clustering", frame, y, "synthetic", 3))
    return _write_cases(root, seed, definitions)


def make_router_cases(root: Path, seed: int) -> list[dict]:
    """New offline families; seed 42 development, seed 8081 held for confirmation.

    The suite generates dense CSV inputs, including mostly-zero feature matrices;
    it does not imply support for sparse file input formats.
    """
    rng = np.random.default_rng(seed)
    iris = load_iris()
    definitions = [("router_iris", "classification", pd.DataFrame(iris.data, columns=iris.feature_names),
                    iris.target, "sklearn bundled real dataset", None)]
    x = rng.normal(size=(720, 100)) * (rng.random((720, 100)) < .12)
    signal = x[:, :8] @ np.arange(1, 9) + rng.normal(0, .4, 720)
    definitions.append(("sparse_linear_classes", "classification", pd.DataFrame(x),
                        (signal > np.median(signal)).astype(int), "synthetic", None))
    x, y = make_classification(n_samples=720, n_features=10, n_informative=5, weights=[.82, .18], random_state=seed)
    frame = pd.DataFrame(x)
    frame[rng.random(frame.shape) < .18] = np.nan
    definitions.append(("missing_interactions", "classification", frame, y, "synthetic", None))
    x = rng.normal(size=(760, 110)) * (rng.random((760, 110)) < .15)
    y = x[:, :12] @ rng.uniform(1, 5, 12) + rng.normal(0, .3, 760)
    definitions.append(("sparse_linear_target", "regression", pd.DataFrame(x), y, "synthetic", None))
    x = rng.normal(size=(760, 9))
    y = 4 * x[:, 0] - 3 * x[:, 1] + rng.standard_t(2, 760)
    definitions.append(("heavy_tail_target", "regression", pd.DataFrame(x), y, "synthetic", None))
    x = rng.uniform(-2, 2, (760, 6))
    y = 3 * x[:, 0] * x[:, 1] + np.sin(3 * x[:, 2]) + rng.normal(0, .2, 760)
    frame = pd.DataFrame(x)
    frame[rng.random(frame.shape) < .12] = np.nan
    definitions.append(("missing_product_target", "regression", frame, y, "synthetic", None))
    x, y = make_blobs(n_samples=[100, 210, 350], centers=[[-6, 0], [0, 5], [5, -3]],
                     cluster_std=[.3, 1.8, .8], random_state=seed)
    definitions.append(("unequal_density", "clustering", pd.DataFrame(x), y, "synthetic", 3))
    x, y = make_blobs(n_samples=660, centers=[[-5, -4], [0, 5], [5, -3]], cluster_std=.8, random_state=seed)
    frame = pd.DataFrame(x)
    frame[rng.random(frame.shape) < .16] = np.nan
    frame["missing_noise"] = rng.normal(size=660)
    frame.loc[rng.random(660) < .5, "missing_noise"] = np.nan
    definitions.append(("missing_cluster_axes", "clustering", frame, y, "synthetic", 3))
    x, y = make_blobs(n_samples=660, centers=3, n_features=3, cluster_std=.9, random_state=seed)
    x = x @ np.array([[1, .8, 0], [0, .3, .1], [.1, 0, 2]])
    definitions.append(("correlated_cluster_axes", "clustering", pd.DataFrame(x), y, "synthetic", 3))
    return _write_cases(root, seed, definitions)


def _write_cases(root, seed, definitions):
    cases = []
    for name, task, frame, target, source, k in definitions:
        frame = frame.copy()
        frame.columns = [str(c) for c in frame.columns]
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        types = {c: "numeric" if pd.api.types.is_numeric_dtype(frame[c]) else "categorical" for c in frame}
        meta = {"task": task, "feature_types": types}
        if task == "clustering":
            train, reference, training_target = frame, np.asarray(target), None
            meta["n_clusters"] = k
        else:
            train_idx, test_idx = train_test_split(np.arange(len(frame)), test_size=.25, random_state=seed,
                                                  stratify=target if task == "classification" else None)
            train = frame.iloc[train_idx].copy()
            training_target, reference = np.asarray(target)[train_idx], np.asarray(target)[test_idx]
            train["target"] = training_target
            frame.iloc[test_idx].to_csv(folder / "test.csv", index=False)
            meta["target"] = "target"
        train.to_csv(folder / "train.csv", index=False)
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        cases.append({"name": name, "family": name, "task": task, "input": folder, "reference": reference,
                      "training_target": training_target, "n_clusters": k, "source": source, "seed": seed,
                      "partition": "confirmation" if seed == 8081 else "development"})
    return cases
