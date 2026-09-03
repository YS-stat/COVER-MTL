"""Audit formal simulation outputs without reading pilot or real-data results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import FORMAL_BLOCKS, SimulationBlock


KEY_METRICS = ("prediction_mse", "excess_mse")


def _normalized_config(path: Path) -> str:
    config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config.get("dgp"), dict):
        dgp = config["dgp"]
        dgp.pop("seed", None)
        dgp.setdefault("rotation_structure", "full")
        dgp.setdefault("common_coordinate_mode", "auto")
    config.setdefault("coupling_selection", "mean")
    # This label is descriptive only; the complete network configuration is
    # compared separately as part of the same JSON object.
    config.pop("network_name", None)
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _config_hash(path: Path) -> str:
    return hashlib.sha256(_normalized_config(path).encode()).hexdigest()[:12]


def _replicate_directories(setting_dir: Path) -> list[Path]:
    direct = sorted(setting_dir.glob("rep_*"))
    if direct:
        return direct
    scenarios = [path for path in setting_dir.iterdir() if path.is_dir()]
    if len(scenarios) != 1:
        raise RuntimeError(
            f"Expected one scenario directory below {setting_dir}; found {scenarios}."
        )
    return sorted(scenarios[0].glob("rep_*"))


def _audit_setting(
    root: Path, block: SimulationBlock, setting: str, *, legacy_layout: bool,
) -> dict[str, object]:
    result_directory = (
        block.legacy_result_directory if legacy_layout else block.result_directory
    )
    setting_dir = root / result_directory / setting
    if not setting_dir.is_dir():
        return {"setting": setting, "complete": False, "error": "missing directory"}
    replicate_dirs = _replicate_directories(setting_dir)
    expected_ids = {f"rep_{index:03d}" for index in range(block.replicates_per_setting)}
    observed_ids = {path.name for path in replicate_dirs}
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)
    failures: list[str] = []
    nonconverged_tuning_rows = 0
    hashes: set[str] = set()
    seeds: list[int] = []
    for replicate_dir in replicate_dirs:
        metrics_path = replicate_dir / "metrics.csv"
        config_path = replicate_dir / "config.json"
        if not metrics_path.is_file() or not config_path.is_file():
            failures.append(f"{replicate_dir.name}: missing metrics.csv or config.json")
            continue
        metrics = pd.read_csv(metrics_path)
        methods = tuple(metrics["method"].astype(str)) if "method" in metrics else ()
        if set(methods) != set(block.methods) or len(methods) != len(block.methods):
            failures.append(f"{replicate_dir.name}: methods={methods}")
        for metric in KEY_METRICS:
            if metric not in metrics or not np.isfinite(metrics[metric]).all():
                failures.append(f"{replicate_dir.name}: invalid {metric}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task_count = int(config.get("dgp", {}).get("num_tasks", 0))
        task_metrics_path = replicate_dir / "task_metrics.csv"
        if not task_metrics_path.is_file():
            failures.append(f"{replicate_dir.name}: missing task_metrics.csv")
        else:
            task_metrics = pd.read_csv(task_metrics_path)
            expected_task_rows = task_count * len(block.methods)
            if len(task_metrics) != expected_task_rows:
                failures.append(
                    f"{replicate_dir.name}: task rows={len(task_metrics)}, "
                    f"expected={expected_task_rows}"
                )
            if "method" not in task_metrics or set(task_metrics["method"]) != set(
                block.methods
            ):
                failures.append(f"{replicate_dir.name}: invalid task-level methods")
            elif "task" not in task_metrics:
                failures.append(f"{replicate_dir.name}: missing task index")
            else:
                expected_tasks = set(range(task_count))
                for method in block.methods:
                    observed_tasks = set(
                        task_metrics.loc[task_metrics["method"] == method, "task"]
                        .astype(int)
                        .tolist()
                    )
                    if observed_tasks != expected_tasks:
                        failures.append(
                            f"{replicate_dir.name}: invalid tasks for {method}"
                        )
            for metric in KEY_METRICS:
                if (
                    metric not in task_metrics
                    or not np.isfinite(task_metrics[metric]).all()
                ):
                    failures.append(
                        f"{replicate_dir.name}: invalid task-level {metric}"
                    )
            if "method" in task_metrics:
                metric_index = metrics.set_index("method")
                for method in block.methods:
                    rows = task_metrics[task_metrics["method"] == method]
                    if method not in metric_index.index or rows.empty:
                        continue
                    for metric in (
                        "prediction_mse",
                        "excess_mse",
                        "common_mse",
                        "deviation_mse",
                        "estimated_deviation_energy",
                    ):
                        if metric not in rows or metric not in metric_index:
                            continue
                        reported = float(metric_index.loc[method, metric])
                        reconstructed = float(rows[metric].mean())
                        if not np.isclose(
                            reported, reconstructed, rtol=1e-10, atol=1e-12
                        ):
                            failures.append(
                                f"{replicate_dir.name}: {method} {metric} "
                                "does not match task rows"
                            )
                    if "worst_task_excess_mse" in metric_index:
                        reported_worst = float(
                            metric_index.loc[method, "worst_task_excess_mse"]
                        )
                        if not np.isclose(
                            reported_worst,
                            float(rows["excess_mse"].max()),
                            rtol=1e-10,
                            atol=1e-12,
                        ):
                            failures.append(
                                f"{replicate_dir.name}: {method} worst-task metric "
                                "does not match task rows"
                            )
        configured_couplings = {
            float(value) for value in config.get("couplings", ())
        } | {0.0}
        coupled = metrics[metrics["method"].isin(("COVER", "Average-Moment"))]
        if (
            not coupled.empty
            and not coupled["selected_coupling"]
            .map(lambda value: float(value) in configured_couplings)
            .all()
        ):
            failures.append(f"{replicate_dir.name}: coupling outside tuning grid")
        tuning_path = replicate_dir / "tuning.csv"
        if tuning_path.is_file() and tuning_path.stat().st_size:
            tuning = pd.read_csv(tuning_path)
            if "converged" in tuning and "method" in tuning:
                classical = tuning[tuning["method"].isin(("ARMUL", "FLARCC"))]
                if not classical.empty:
                    converged = (
                        classical["converged"].astype(str).str.lower().eq("true")
                    )
                    nonconverged_tuning_rows += int((~converged).sum())
        hashes.add(_config_hash(config_path))
        seed = config.get("dgp", {}).get("seed")
        if isinstance(seed, int):
            seeds.append(seed)
        log_path = replicate_dir / "run.log"
        if log_path.is_file() and log_path.stat().st_size:
            failures.append(f"{replicate_dir.name}: nonempty run.log")
    expected_seed_count = (
        1 if block.seed_policy == "fixed" else block.replicates_per_setting
    )
    seed_count_ok = len(set(seeds)) == expected_seed_count
    config_count_ok = len(hashes) == 1
    complete = (
        not missing_ids
        and not unexpected_ids
        and not failures
        and seed_count_ok
        and config_count_ok
    )
    return {
        "setting": setting,
        "replicates": len(replicate_dirs),
        "missing_replicates": missing_ids,
        "unexpected_replicates": unexpected_ids,
        "config_hashes": sorted(hashes),
        "unique_seeds": len(set(seeds)),
        "expected_unique_seeds": expected_seed_count,
        "failures": failures,
        "nonconverged_classical_tuning_rows": nonconverged_tuning_rows,
        "complete": complete,
    }


def audit(results_root: Path, *, legacy_layout: bool = False) -> dict[str, object]:
    blocks = []
    for block in FORMAL_BLOCKS:
        settings = [
            _audit_setting(results_root, block, setting, legacy_layout=legacy_layout,)
            for setting in block.settings
        ]
        result_directory = (
            block.legacy_result_directory if legacy_layout else block.result_directory
        )
        blocks.append(
            {
                "name": block.name,
                "result_directory": result_directory,
                "complete": all(bool(item["complete"]) for item in settings),
                "settings": settings,
            }
        )
    theory_path = results_root / "theory_verification" / "fixed_representation.csv"
    theory: dict[str, object] = {"complete": False, "path": str(theory_path)}
    if theory_path.is_file():
        frame = pd.read_csv(theory_path)
        theory = {
            "complete": bool(
                len(frame) == 155
                and frame["design"].nunique() == 5
                and frame.groupby("design")["coupling"].nunique().eq(31).all()
                and np.isfinite(frame["relative_error"]).all()
            ),
            "rows": len(frame),
            "designs": int(frame["design"].nunique()),
            "couplings_per_design": sorted(
                frame.groupby("design")["coupling"].nunique().unique().tolist()
            ),
            "maximum_relative_error": float(frame["relative_error"].max()),
        }
    return {
        "complete": all(bool(block["complete"]) for block in blocks)
        and bool(theory["complete"]),
        "blocks": blocks,
        "fixed_representation_theory": theory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="Audit the historical server directory names used for the revision runs.",
    )
    args = parser.parse_args()
    report = audit(args.results_root, legacy_layout=args.legacy_layout)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
