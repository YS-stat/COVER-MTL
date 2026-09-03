"""Aggregate formal simulation suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_METRICS = (
    "prediction_mse",
    "excess_mse",
    "worst_task_excess_mse",
    "common_mse",
    "deviation_mse",
    "estimated_deviation_energy",
    "fraction_tasks_improved_vs_stl",
    "fit_seconds",
    "workflow_seconds",
    "peak_device_memory_mb",
    "parameter_count",
    "auxiliary_parameter_count",
)


def _standard_error(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    if array.size <= 1:
        return float("nan")
    return float(array.std(ddof=1) / np.sqrt(array.size))


def _read_with_setting(path: Path, input_dir: Path, suite: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    relative = path.relative_to(input_dir)
    if suite == "main" and len(relative.parts) >= 4:
        frame["setting"] = "/".join(relative.parts[:2])
    else:
        frame["setting"] = relative.parts[0]
    config_path = path.with_name("config.json")
    if path.name == "metrics.csv" and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        task_count = int(config["dgp"]["num_tasks"])
        representation_dim = int(config["network"]["representation_dim"])
        frame["num_tasks"] = task_count
        frame["input_dim"] = int(config["dgp"]["input_dim"])
        frame["representation_dim"] = representation_dim
        if "parameter_count" in frame:
            legacy_counts = int(config.get("output_schema_version", 1)) < 2
            fixed_head = frame["method"].isin(("ARMUL", "FLARCC"))
            if legacy_counts:
                frame.loc[fixed_head, "parameter_count"] += (
                    task_count * representation_dim
                )
            cover_auxiliary = (frame["method"] == "COVER") & (
                frame["selected_coupling"] > 0
            )
            auxiliary_count = task_count * (task_count - 1) // 2 * representation_dim
            if legacy_counts:
                frame.loc[cover_auxiliary, "parameter_count"] -= auxiliary_count
            frame["auxiliary_parameter_count"] = 0
            frame.loc[cover_auxiliary, "auxiliary_parameter_count"] = auxiliary_count
    return frame


def _add_workflow_seconds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Include required HPS initialization in dependent-method runtimes."""
    if "fit_seconds" not in metrics:
        return metrics
    result = metrics.copy()
    if "workflow_seconds" not in result:
        result["workflow_seconds"] = result["fit_seconds"]
    else:
        result["workflow_seconds"] = result["workflow_seconds"].fillna(
            result["fit_seconds"]
        )
    replicate_column = "repeat" if "repeat" in result else "replicate"
    hps = result[result["method"] == "HPS"].set_index(["setting", replicate_column])[
        "fit_seconds"
    ]
    dependent = result["method"].isin(("Average-Moment", "COVER", "ARMUL", "FLARCC"))
    missing_workflow = dependent & ~result["workflow_seconds"].gt(
        result["fit_seconds"] + 1e-12
    )
    if missing_workflow.any():
        missing_keys = pd.MultiIndex.from_frame(
            result.loc[missing_workflow, ["setting", replicate_column]]
        )
        result.loc[missing_workflow, "workflow_seconds"] += hps.reindex(
            missing_keys
        ).to_numpy()
    return result


def _summarize(metrics: pd.DataFrame, suite: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (setting, method), frame in metrics.groupby(["setting", "method"]):
        replicate_column = "repeat" if "repeat" in frame else "replicate"
        row: dict[str, object] = {
            "setting": setting,
            "method": method,
            "replicates": int(frame[replicate_column].nunique()),
        }
        for metric in SUMMARY_METRICS:
            if metric not in frame:
                continue
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_se"] = _standard_error(frame[metric])
        if "selected_coupling" in frame:
            selected = frame["selected_coupling"].dropna()
            row["median_selected_coupling"] = (
                float(selected.median()) if not selected.empty else float("nan")
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    rank_metric = "prediction_mse_mean"
    if "excess_mse_mean" in summary and summary["excess_mse_mean"].notna().any():
        rank_metric = "excess_mse_mean"
    return summary.sort_values(["setting", rank_metric])


def _paired_cover_comparisons(metrics: pd.DataFrame, suite: str) -> pd.DataFrame:
    replicate_column = "repeat" if "repeat" in metrics else "replicate"
    candidates = (
        "prediction_mse",
        "excess_mse",
        "worst_task_excess_mse",
        "common_mse",
        "deviation_mse",
    )
    metric_names = [name for name in candidates if name in metrics]
    rows: list[dict[str, object]] = []
    for setting, setting_frame in metrics.groupby("setting"):
        cover = setting_frame.loc[setting_frame["method"] == "COVER"].set_index(
            replicate_column
        )
        for method in sorted(set(setting_frame["method"]) - {"COVER"}):
            other = setting_frame.loc[setting_frame["method"] == method].set_index(
                replicate_column
            )
            common = cover.index.intersection(other.index)
            for metric in metric_names:
                difference = cover.loc[common, metric] - other.loc[common, metric]
                relative = -difference / other.loc[common, metric]
                rows.append(
                    {
                        "setting": setting,
                        "comparison": f"COVER - {method}",
                        "metric": metric,
                        "replicates": int(len(common)),
                        "mean_difference": float(difference.mean()),
                        "standard_error": _standard_error(difference),
                        "mean_relative_improvement": float(relative.mean()),
                        "cover_win_fraction": float((difference < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", required=True, choices=("main", "scaling", "sensitivity")
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-replicates", type=int, default=None)
    parser.add_argument("--expected-settings", default=None)
    args = parser.parse_args()

    metric_files = sorted(args.input_dir.glob("**/metrics.csv"))
    task_files = sorted(args.input_dir.glob("**/task_metrics.csv"))
    tuning_files = sorted(args.input_dir.glob("**/tuning.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics found below {args.input_dir}.")

    metrics = pd.concat(
        [_read_with_setting(path, args.input_dir, args.suite) for path in metric_files],
        ignore_index=True,
    )
    metrics = _add_workflow_seconds(metrics)
    tasks = (
        pd.concat(
            [
                _read_with_setting(path, args.input_dir, args.suite)
                for path in task_files
            ],
            ignore_index=True,
        )
        if task_files
        else pd.DataFrame()
    )
    tuning = (
        pd.concat(
            [
                _read_with_setting(path, args.input_dir, args.suite)
                for path in tuning_files
            ],
            ignore_index=True,
        )
        if tuning_files
        else pd.DataFrame()
    )

    summary = _summarize(metrics, args.suite)
    paired = _paired_cover_comparisons(metrics, args.suite)
    selection = pd.DataFrame()
    if "selected_coupling" in metrics:
        selection = (
            metrics.groupby(["setting", "method", "selected_coupling"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )

    replicate_column = "repeat" if "repeat" in metrics else "replicate"
    counts = (
        metrics.groupby(["setting", "method"])[replicate_column]
        .nunique()
        .rename("replicates")
        .reset_index()
    )
    expected = args.expected_replicates
    expected_settings = (
        tuple(value for value in args.expected_settings.split(",") if value)
        if args.expected_settings
        else tuple()
    )
    observed_settings = set(metrics["setting"].unique())
    settings_complete = (
        not expected_settings or set(expected_settings) == observed_settings
    )
    audit = {
        "suite": args.suite,
        "metric_files": len(metric_files),
        "task_files": len(task_files),
        "tuning_files": len(tuning_files),
        "settings": sorted(metrics["setting"].unique().tolist()),
        "methods": sorted(metrics["method"].unique().tolist()),
        "expected_replicates": expected,
        "expected_settings": list(expected_settings),
        "complete": bool(
            settings_complete
            and (expected is None or (counts["replicates"] == expected).all())
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "all_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "metric_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    counts.to_csv(args.output_dir / "completion_counts.csv", index=False)
    if not tasks.empty:
        tasks.to_csv(args.output_dir / "all_task_metrics.csv", index=False)
    if not tuning.empty:
        tuning.to_csv(args.output_dir / "all_tuning.csv", index=False)
    if not selection.empty:
        selection.to_csv(args.output_dir / "selection_summary.csv", index=False)
    with (args.output_dir / "aggregation_audit.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(audit, stream, indent=2)

    columns = ["setting", "method", "replicates"]
    display_metrics = ("excess_mse", "prediction_mse", "workflow_seconds")
    for metric in display_metrics:
        if f"{metric}_mean" in summary:
            columns.extend([f"{metric}_mean", f"{metric}_se"])
            break
    print(summary[columns].to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
