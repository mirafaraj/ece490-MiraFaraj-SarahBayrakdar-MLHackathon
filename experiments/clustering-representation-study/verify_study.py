"""Independent profile reconstruction, exact repeats and small actual-CLI checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/reliability-preparation"), str(HERE)]


def main():
    import numpy as np
    from compare import POLICIES, save, sha
    from profile_reference import reconstruct, compare
    from unknown_dataset.clustering import fit_predict_clustering
    from unknown_dataset.contract import load_dataset
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert all(sha(ROOT / p) == h for p, h in protocol["source_sha256"].items())
    fields = 0
    for case in protocol["cases"]:
        source = HERE / "local/inputs" / case["id"]
        actual = json.loads((HERE / "local/profiles" / (case["id"] + ".json")).read_text())
        fields += compare(reconstruct(source), actual)
    repeat_rows = []
    ids = ["mask_only-rare-104743", "nuisance_blocks-balanced-104729"]
    for identity in ids:
        case = next(c for c in protocol["cases"] if c["id"] == identity)
        data = load_dataset(HERE / "local/inputs" / identity)
        for policy, controls in POLICIES.items():
            pred, info = fit_predict_clustering("clustering", data.X_train, feature_types=data.feature_types,
                n_clusters=case["k"], seed=case["seed"], time_budget=20, experimental_controls=controls)
            old = json.loads((HERE / "local/predictions" / identity / (policy + ".json")).read_text())
            np.testing.assert_array_equal(pred, old)
            repeat_rows.append(dict(id=identity, policy=policy, identical_predictions=True, selected=info["model"]))
    cli_rows = []
    for identity in ids + ["mall-random30"]:
        case = next(c for c in protocol["cases"] if c["id"] == identity)
        for policy in (list(POLICIES) if identity != "mall-random30" else ["submission"]):
            source = HERE / "local/inputs" / identity
            output = HERE / "local/cli" / identity / policy
            command = [sys.executable, str(ROOT / "main.py"), "--input", str(source), "--output", str(output),
                       "--time-budget", "50", "--seed", str(case["seed"]), "--strategy"]
            if policy == "submission":
                command += ["submission"]
            else:
                config = HERE / "local/configs" / (policy + ".json")
                if not config.exists():
                    save(config, {"experimental": {**POLICIES[policy], "runtime_instrumentation": True}})
                command += ["routed", "--router-config", str(config), "--timings"]
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            start = time.monotonic()
            completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            wall = time.monotonic() - start
            log = HERE / "local/cli" / identity / (policy + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(completed.stdout + completed.stderr)
            assert completed.returncode == 0, (identity, policy, completed.stderr)
            run_info = json.loads((output / "run.json").read_text())
            model = run_info["model"]
            assert wall < 60 and not model.get("fallback"), run_info
            import pandas as pd
            prediction = pd.read_csv(output / "predictions.csv").prediction.to_numpy()
            assert len(prediction) == case["rows"] and set(prediction) == set(range(case["k"]))
            paired_policy = "full" if policy == "submission" else policy
            direct = json.loads((HERE / "local/predictions" / identity / (paired_policy + ".json")).read_text())
            np.testing.assert_array_equal(prediction, direct)
            compare(reconstruct(source), json.loads((output / "profile.json").read_text()))
            cli_rows.append(dict(id=identity, policy=policy, seconds=wall, valid=True,
                selected=model["model"], fallback=model.get("fallback", False),
                recovery=model.get("fallback_kind", "none"), route_stage=model.get("route_stage"),
                matches_direct=True, predictions_sha256=sha(output / "predictions.csv")))
            print(identity, policy, round(wall, 3), "CLI verified", flush=True)
    assert all(sha(ROOT / p) == h for p, h in protocol["source_sha256"].items())
    save(HERE / "verification.json", dict(profile_cases=len(protocol["cases"]), profile_fields=fields,
        float_tolerance=dict(rtol=1e-10, atol=1e-12), repeat_rows=repeat_rows, cli_rows=cli_rows,
        source_hashes_unchanged=True, memory_evidence="Per-case peak memory unavailable; not inferred from runtime."))


if __name__ == "__main__":
    main()
