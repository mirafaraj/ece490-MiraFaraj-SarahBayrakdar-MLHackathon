"""Budgeted CLI orchestration with an isolated, terminable model worker."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
import traceback
from pathlib import Path


ARTIFACT_NAMES = ("predictions.csv", "profile.json", "run.json")


def _clear_artifacts(directory):
    """Remove only files owned by this entry point, never input or user files."""
    for name in ARTIFACT_NAMES:
        (directory / name).unlink(missing_ok=True)


def _model_worker(dataset, budget, seed, result_path, strategy, route_config=None, checkpoint_path=None):
    try:
        from .models import fit_predict

        extra = {}
        if strategy == "routed":
            from .checkpoints import write_checkpoint

            extra = {"route_config": route_config,
                     "checkpoint_callback": lambda predictions, info: write_checkpoint(
                         checkpoint_path, dataset, predictions, info)}
        predictions, info = fit_predict(
            task=dataset.task,
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_test=dataset.X_test,
            feature_types=dataset.feature_types,
            n_clusters=dataset.n_clusters,
            time_budget=budget,
            seed=seed,
            strategy=strategy,
            **extra,
        )
        payload = {"predictions": predictions, "model_info": info}
    except Exception:
        payload = {"error": traceback.format_exc()}
    # Files avoid pipe deadlocks when a prediction vector exceeds the pipe buffer.
    staging_path = result_path + ".tmp"
    with open(staging_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(staging_path, result_path)


def _recover_checkpoint(checkpoint_path, dataset, reason):
    from .checkpoints import load_checkpoint

    try:
        result = load_checkpoint(checkpoint_path, dataset)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    result["model_info"] = dict(result["model_info"], fallback=True,
                                fallback_kind="completed_baseline", checkpoint_recovered=True,
                                fallback_reason=str(reason))
    return result


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-budget", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("baseline", "auto", "routed"), default="baseline")
    parser.add_argument("--router-config", type=Path)
    args = parser.parse_args(argv)
    if not 0 < args.time_budget <= 3600:
        parser.error("--time-budget must be positive and at most 3600 seconds")
    if args.router_config is not None and args.strategy != "routed":
        parser.error("--router-config requires --strategy routed")

    process = None
    try:
        _clear_artifacts(args.output)
        from .contract import load_dataset, write_outputs
        from .profiling import build_profile
        from .checkpoints import validate_predictions

        route_config = None
        if args.strategy == "routed":
            from .routing import load_router_config

            route_config = load_router_config(args.router_config)

        dataset = load_dataset(args.input)
        profile = build_profile(dataset.X_train, dataset.feature_types, dataset.y_train)
        reserve = min(3.0, args.time_budget * 0.15)
        remaining = args.time_budget - (time.monotonic() - started) - reserve
        if remaining <= 0:
            raise TimeoutError("Input loading and profiling exhausted the execution budget")

        args.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".worker-", dir=args.output) as temporary:
            result_path = str(Path(temporary) / "model-result.pkl")
            checkpoint_path = str(Path(temporary) / "completed-baseline.json")
            worker_args = (dataset, remaining, args.seed, result_path, args.strategy)
            if args.strategy == "routed":
                worker_args += (route_config, checkpoint_path)
            process = mp.get_context("spawn").Process(
                target=_model_worker,
                args=worker_args,
            )
            process.start()
            wait_budget = max(0, args.time_budget - (time.monotonic() - started) - reserve)
            process.join(wait_budget)
            if process.is_alive():
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
                result = (_recover_checkpoint(checkpoint_path, dataset, "model worker deadline")
                          if args.strategy == "routed" else None)
                if result is None:
                    from .fallback import fallback_predictions

                    predictions, info = fallback_predictions(
                        task=dataset.task, X_train=dataset.X_train,
                        y_train=dataset.y_train, X_test=dataset.X_test,
                        n_clusters=dataset.n_clusters, seed=args.seed,
                        reason="model worker deadline",
                    )
                    result = {"predictions": predictions, "model_info": info}
            else:
                try:
                    if process.exitcode != 0 or not Path(result_path).exists():
                        raise RuntimeError(f"Model worker exited without a result (exit={process.exitcode})")
                    # Only this invocation's locally created worker result is unpickled.
                    with open(result_path, "rb") as handle:
                        result = pickle.load(handle)
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    validate_predictions(dataset, result["predictions"])
                except Exception as exc:
                    result = (_recover_checkpoint(checkpoint_path, dataset, str(exc))
                              if args.strategy == "routed" else None)
                    if result is None:
                        raise

        predictions = validate_predictions(dataset, result["predictions"])
        run_info = {
            "task": dataset.task,
            "seed": args.seed,
            "time_budget_seconds": args.time_budget,
            "elapsed_before_output_seconds": time.monotonic() - started,
            "model": result["model_info"],
            "contract": "provisional-v1",
            "strategy": args.strategy,
        }
        write_outputs(args.output, predictions, profile, run_info)
        elapsed = time.monotonic() - started
        if elapsed > args.time_budget:
            raise TimeoutError("Output serialization exceeded the execution budget")
        print(json.dumps({"status": "ok", "elapsed_seconds": elapsed}), file=sys.stderr)
        return 0
    except Exception as exc:
        try:
            _clear_artifacts(args.output)
        except OSError:
            pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if process is not None and process.is_alive():
            process.terminate()
            process.join()
