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


def _model_worker(dataset, budget, seed, result_path, strategy, route_config=None, checkpoint_path=None,
                  instrument_runtime=False, execution_deadline=None, reliability_mode=None):
    worker_started = time.monotonic()
    try:
        from .models import fit_predict

        extra = {}
        if reliability_mode is not None:
            extra['reliability_mode'] = reliability_mode
        if strategy == "routed":
            from .checkpoints import write_checkpoint

            extra.update({"route_config": route_config,
                     "checkpoint_callback": lambda predictions, info: write_checkpoint(
                         checkpoint_path, dataset, predictions, info)})
            if execution_deadline is not None:
                extra["execution_deadline"] = execution_deadline
        fit_started = time.monotonic()
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
        if instrument_runtime:
            payload["worker_phase_seconds"] = {
                "imports": fit_started - worker_started,
                "fit_predict": time.monotonic() - fit_started,
            }
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


def _bounded_worker_result(dataset, seed, strategy, route_config, temporary, deadline, instrument):
    """One fixed forest attempt, then existing linear recovery within one deadline."""
    from .checkpoints import validate_predictions
    attempts = []
    checkpoint = str(Path(temporary) / 'completed-bounded.json')
    for mode in ('forest', 'linear'):
        available = deadline - time.monotonic()
        if available <= 0:
            break
        result_path = str(Path(temporary) / (mode + '.pkl'))
        allowance = min(25.0, available * .65) if mode == 'forest' else available
        process = mp.get_context('spawn').Process(target=_model_worker,
            args=(dataset, available, seed, result_path, strategy, route_config,
                  checkpoint, instrument, deadline, mode))
        started = time.monotonic()
        reason = None
        try:
            process.start()
            # Includes process.start's input transfer, rather than adding time after it.
            process.join(max(0, min(deadline, started + allowance) - time.monotonic()))
            if process.is_alive():
                reason = mode + ' worker deadline'
            elif process.exitcode != 0 or not Path(result_path).exists():
                reason = mode + ' worker exited without a result'
            else:
                with open(result_path, 'rb') as handle:
                    result = pickle.load(handle)
                if 'error' in result:
                    reason = result['error']
                else:
                    validate_predictions(dataset, result['predictions'])
                    attempts.append(dict(mode=mode, seconds=time.monotonic()-started, outcome='completed'))
                    result['model_info']['reliability_attempts'] = attempts
                    return result
        except Exception as exc:
            reason = f'{mode} result failure: {type(exc).__name__}: {exc}'
        finally:
            if process.is_alive():
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
        attempts.append(dict(mode=mode, seconds=time.monotonic()-started, outcome='failed', reason=reason))
        recovered = _recover_checkpoint(checkpoint, dataset, reason)
        if recovered is not None:
            recovered['model_info']['reliability_attempts'] = attempts
            return recovered
    from .fallback import fallback_predictions
    predictions, info = fallback_predictions(dataset.task, dataset.X_train, dataset.y_train,
        dataset.X_test, n_clusters=dataset.n_clusters, seed=seed,
        reason='bounded forest and linear recovery unavailable')
    info['reliability_attempts'] = attempts
    return dict(predictions=predictions, model_info=info)


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-budget", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=("baseline", "auto", "routed", "submission"), default="submission")
    parser.add_argument("--router-config", type=Path)
    parser.add_argument("--timings", action="store_true", help="Opt in to diagnostic phase timings")
    parser.add_argument("--lazy-router-imports", action="store_true",
                        help="Experimental: load routed configuration without parent modeling imports")
    args = parser.parse_args(argv)
    if not 0 < args.time_budget <= 3600:
        parser.error("--time-budget must be positive and at most 3600 seconds")
    if args.router_config is not None and args.strategy != "routed":
        parser.error("--router-config requires --strategy routed")
    if args.lazy_router_imports and args.strategy != "routed":
        parser.error("--lazy-router-imports requires --strategy routed")

    process = None
    phases = {}
    try:
        _clear_artifacts(args.output)
        stage_started = time.monotonic()
        from .contract import load_dataset, write_outputs
        from .profiling import build_profile
        from .checkpoints import validate_predictions
        if args.timings:
            phases["parent_data_imports"] = time.monotonic() - stage_started

        route_config = None
        if args.strategy == "routed":
            stage_started = time.monotonic()
            if args.lazy_router_imports:
                from .lightweight_config import load_router_config
            else:
                from .routing import load_router_config

            if args.timings:
                phases["parent_router_imports"] = time.monotonic() - stage_started

            stage_started = time.monotonic()
            route_config = load_router_config(args.router_config)
            if args.timings:
                phases["parent_router_configuration"] = time.monotonic() - stage_started
                route_config["experimental"]["runtime_instrumentation"] = True

        stage_started = time.monotonic()
        dataset = load_dataset(args.input)
        phases["load"] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        profile = build_profile(dataset.X_train, dataset.feature_types, dataset.y_train,
                                n_clusters=dataset.n_clusters)
        phases["profile"] = time.monotonic() - stage_started
        from .reliability import needs_bounded_forest
        bounded = needs_bounded_forest(dataset.task, len(dataset.X_train), dataset.X_train.shape[1], args.strategy,
                                       dataset.X_train, dataset.feature_types)
        reserve = min(8.0, args.time_budget * .2) if bounded else min(3.0, args.time_budget * 0.15)
        remaining = args.time_budget - (time.monotonic() - started) - reserve
        if remaining <= 0:
            raise TimeoutError("Input loading and profiling exhausted the execution budget")

        args.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".worker-", dir=args.output) as temporary:
            result_path = str(Path(temporary) / "model-result.pkl")
            checkpoint_path = str(Path(temporary) / "completed-baseline.json")
            if bounded:
                stage_started = time.monotonic()
                result = _bounded_worker_result(dataset, args.seed, args.strategy, route_config,
                    temporary, started + args.time_budget - reserve, args.timings)
                phases['bounded_attempts_and_recovery'] = time.monotonic() - stage_started
            else:
                worker_args = (dataset, remaining, args.seed, result_path, args.strategy)
                if args.strategy == "routed" or args.timings:
                    worker_args += (route_config, checkpoint_path, args.timings)
                if args.strategy == "routed" and route_config["experimental"].get("refit_admission", False):
                    # Same monotonic deadline used by parent enforcement. Includes
                    # spawn, unpickling and imports, unlike model-local elapsed time.
                    worker_args += (started + args.time_budget - reserve,)
                process = mp.get_context("spawn").Process(
                    target=_model_worker,
                    args=worker_args,
                )
                stage_started = time.monotonic()
                process.start()
                wait_budget = max(0, args.time_budget - (time.monotonic() - started) - reserve)
                process.join(wait_budget)
                phases["spawn_and_wait"] = time.monotonic() - stage_started
                stage_started = time.monotonic()
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
                phases["recovery_or_result_validation"] = time.monotonic() - stage_started

        predictions = validate_predictions(dataset, result["predictions"])
        run_info = {
            "task": dataset.task,
            "seed": args.seed,
            "time_budget_seconds": args.time_budget,
            "elapsed_before_output_seconds": time.monotonic() - started,
            "model": result["model_info"],
            "contract": "provisional-v1",
            "strategy": args.strategy,
            "numeric_feature_coercions": dataset.numeric_feature_coercions,
        }
        if args.timings:
            run_info["phase_seconds"] = dict(phases)
            run_info["worker_phase_seconds"] = result.get("worker_phase_seconds")
        stage_started = time.monotonic()
        write_outputs(args.output, predictions, profile, run_info)
        output_seconds = time.monotonic() - stage_started
        elapsed = time.monotonic() - started
        if elapsed > args.time_budget:
            raise TimeoutError("Output serialization exceeded the execution budget")
        status = {"status": "ok", "elapsed_seconds": elapsed}
        if args.timings:
            status["output_serialization_seconds"] = output_seconds
        print(json.dumps(status), file=sys.stderr)
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
