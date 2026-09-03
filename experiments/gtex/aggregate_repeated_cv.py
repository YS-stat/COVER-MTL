"""Aggregate repeated donor-level GTEx CV and pairwise COVER diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("COVER", "ARMUL", "FLARCC", "HPS", "Pool", "STL", "MMoE")
MEAN_BASELINES = ("COVER", "TissueMean", "GlobalMean")


def mean_sd(frame: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    return frame.groupby(groups, as_index=False)[values].agg(["mean", "std"]).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    expected = args.repeats * 2 * 5

    neural_paths = sorted(args.input_dir.glob("repeat_*/*/fold_*/metrics.csv"))
    classical_paths = sorted(
        args.input_dir.glob("repeat_*/*/fold_*/classical_metrics.csv")
    )
    overlap_paths = sorted(args.input_dir.glob("repeat_*/*/fold_*/overlap.csv"))
    if len(neural_paths) != expected or len(classical_paths) != expected:
        raise ValueError(
            "Incomplete repeated CV: "
            f"expected {expected} neural and classical files, found "
            f"{len(neural_paths)} and {len(classical_paths)}."
        )
    if len(overlap_paths) != expected:
        raise ValueError(f"Expected {expected} overlap files, found {len(overlap_paths)}.")

    neural_metrics = pd.concat(
        [pd.read_csv(path) for path in neural_paths], ignore_index=True
    )
    classical_metrics = pd.concat(
        [pd.read_csv(path) for path in classical_paths], ignore_index=True
    )
    metrics = pd.concat([neural_metrics, classical_metrics], ignore_index=True)
    metrics = metrics.loc[metrics["method"].isin(METHODS)].copy()
    counts = metrics.groupby("method")["cv_repeat"].nunique()
    if set(counts.index) != set(METHODS) or not (counts == args.repeats).all():
        raise ValueError(f"Method completion audit failed:\n{counts}")

    metrics["standardized_sse"] = metrics["standardized_mse"] * metrics["test"]
    metrics["raw_sse"] = metrics["raw_mse"] * metrics["test"]
    per_tissue = (
        metrics.groupby(
            ["cv_repeat", "response", "method", "tissue"], as_index=False
        )
        .agg(
            standardized_sse=("standardized_sse", "sum"),
            raw_sse=("raw_sse", "sum"),
            test_samples=("test", "sum"),
        )
    )
    per_tissue["standardized_mse"] = (
        per_tissue["standardized_sse"] / per_tissue["test_samples"]
    )
    per_tissue["raw_mse"] = per_tissue["raw_sse"] / per_tissue["test_samples"]
    per_repeat_response = (
        per_tissue.groupby(["cv_repeat", "response", "method"], as_index=False)
        .agg(
            task_balanced_standardized_mse=("standardized_mse", "mean"),
            task_balanced_raw_mse=("raw_mse", "mean"),
            tissues=("tissue", "nunique"),
        )
    )
    per_repeat_overall = (
        per_repeat_response.groupby(["cv_repeat", "method"], as_index=False)
        .agg(
            equal_response_standardized_mse=(
                "task_balanced_standardized_mse",
                "mean",
            ),
            equal_response_raw_mse=("task_balanced_raw_mse", "mean"),
            responses=("response", "nunique"),
        )
    )

    response_summary = (
        per_repeat_response.groupby(["response", "method"], as_index=False)
        .agg(
            standardized_mse_mean=("task_balanced_standardized_mse", "mean"),
            standardized_mse_sd=("task_balanced_standardized_mse", "std"),
            raw_mse_mean=("task_balanced_raw_mse", "mean"),
            raw_mse_sd=("task_balanced_raw_mse", "std"),
        )
        .sort_values(["response", "standardized_mse_mean"])
    )
    overall_summary = (
        per_repeat_overall.groupby("method", as_index=False)
        .agg(
            standardized_mse_mean=("equal_response_standardized_mse", "mean"),
            standardized_mse_sd=("equal_response_standardized_mse", "std"),
            raw_mse_mean=("equal_response_raw_mse", "mean"),
            raw_mse_sd=("equal_response_raw_mse", "std"),
        )
        .sort_values("standardized_mse_mean")
    )
    response_summary["standardized_mean_sd"] = response_summary.apply(
        lambda row: f"{row.standardized_mse_mean:.4f} ({row.standardized_mse_sd:.4f})",
        axis=1,
    )
    overall_summary["standardized_mean_sd"] = overall_summary.apply(
        lambda row: f"{row.standardized_mse_mean:.4f} ({row.standardized_mse_sd:.4f})",
        axis=1,
    )

    baseline_metrics = neural_metrics.loc[
        neural_metrics["method"].isin(MEAN_BASELINES)
    ].copy()
    baseline_metrics["standardized_sse"] = (
        baseline_metrics["standardized_mse"] * baseline_metrics["test"]
    )
    baseline_metrics["raw_sse"] = (
        baseline_metrics["raw_mse"] * baseline_metrics["test"]
    )
    baseline_per_tissue = (
        baseline_metrics.groupby(
            ["cv_repeat", "response", "method", "tissue"], as_index=False
        )
        .agg(
            standardized_sse=("standardized_sse", "sum"),
            raw_sse=("raw_sse", "sum"),
            test_samples=("test", "sum"),
        )
    )
    baseline_per_tissue["standardized_mse"] = (
        baseline_per_tissue["standardized_sse"]
        / baseline_per_tissue["test_samples"]
    )
    baseline_per_tissue["raw_mse"] = (
        baseline_per_tissue["raw_sse"] / baseline_per_tissue["test_samples"]
    )
    baseline_per_response = (
        baseline_per_tissue.groupby(
            ["cv_repeat", "response", "method"], as_index=False
        )
        .agg(
            task_balanced_standardized_mse=("standardized_mse", "mean"),
            task_balanced_raw_mse=("raw_mse", "mean"),
        )
    )
    baseline_per_overall = (
        baseline_per_response.groupby(["cv_repeat", "method"], as_index=False)
        .agg(
            equal_response_standardized_mse=(
                "task_balanced_standardized_mse",
                "mean",
            ),
            equal_response_raw_mse=("task_balanced_raw_mse", "mean"),
        )
    )
    baseline_response_summary = (
        baseline_per_response.groupby(["response", "method"], as_index=False)
        .agg(
            standardized_mse_mean=("task_balanced_standardized_mse", "mean"),
            standardized_mse_sd=("task_balanced_standardized_mse", "std"),
            raw_mse_mean=("task_balanced_raw_mse", "mean"),
            raw_mse_sd=("task_balanced_raw_mse", "std"),
        )
        .sort_values(["response", "standardized_mse_mean"])
    )
    baseline_overall_summary = (
        baseline_per_overall.groupby("method", as_index=False)
        .agg(
            standardized_mse_mean=("equal_response_standardized_mse", "mean"),
            standardized_mse_sd=("equal_response_standardized_mse", "std"),
            raw_mse_mean=("equal_response_raw_mse", "mean"),
            raw_mse_sd=("equal_response_raw_mse", "std"),
        )
        .sort_values("standardized_mse_mean")
    )
    baseline_task_means = baseline_per_tissue.groupby(
        ["response", "tissue", "method"]
    )["standardized_mse"].mean().unstack()
    baseline_repeat_means = baseline_per_overall.pivot(
        index="cv_repeat",
        columns="method",
        values="equal_response_standardized_mse",
    )
    mean_baseline_audit = {}
    for baseline in ("TissueMean", "GlobalMean"):
        mean_baseline_audit[baseline] = {
            "cover_task_wins": int(
                (baseline_task_means["COVER"] < baseline_task_means[baseline]).sum()
            ),
            "response_tissue_combinations": int(len(baseline_task_means)),
            "cover_repeat_wins": int(
                (baseline_repeat_means["COVER"] < baseline_repeat_means[baseline]).sum()
            ),
            "repeats": args.repeats,
            "mean_relative_gain": float(
                (
                    1.0
                    - baseline_repeat_means["COVER"]
                    / baseline_repeat_means[baseline]
                ).mean()
            ),
        }

    pivot = per_repeat_overall.pivot(
        index="cv_repeat", columns="method", values="equal_response_standardized_mse"
    )
    gain_rows = []
    for method in METHODS:
        if method == "COVER":
            continue
        difference = pivot[method] - pivot["COVER"]
        relative = 1.0 - pivot["COVER"] / pivot[method]
        gain_rows.append(
            {
                "comparator": method,
                "mse_difference_mean": float(difference.mean()),
                "mse_difference_sd": float(difference.std(ddof=1)),
                "relative_gain_mean": float(relative.mean()),
                "relative_gain_sd": float(relative.std(ddof=1)),
                "cover_win_fraction": float((difference > 0).mean()),
            }
        )
    paired_gains = pd.DataFrame(gain_rows).sort_values("relative_gain_mean", ascending=False)
    repeat_winner = pivot.idxmin(axis=1).rename("winner").reset_index()
    win_counts = (
        repeat_winner.groupby("winner", as_index=False)
        .agg(repetitions_won=("cv_repeat", "size"))
        .sort_values("repetitions_won", ascending=False)
    )

    fold_runtime = (
        metrics.groupby(
            ["cv_repeat", "response", "test_fold", "method"], as_index=False
        )["fit_seconds"]
        .first()
    )
    repeat_runtime = (
        fold_runtime.groupby(["cv_repeat", "method"], as_index=False)["fit_seconds"]
        .sum()
    )
    runtime_summary = (
        repeat_runtime.groupby("method", as_index=False)["fit_seconds"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "fit_seconds_mean", "std": "fit_seconds_sd"})
    )

    overlaps = pd.concat([pd.read_csv(path) for path in overlap_paths], ignore_index=True)
    pair_keys = [
        "cv_repeat",
        "response",
        "left_tissue",
        "right_tissue",
    ]
    diagnostic_columns = [
        "normalized_overlap_trace",
        "overlap_effective_rank",
        "supported_contrast_fraction",
        "overlap_weighted_head_energy",
        "pairwise_penalty_contribution",
    ]
    eigenvalue_columns = sorted(
        [column for column in overlaps if column.startswith("overlap_eigenvalue_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    per_repeat_pair = (
        overlaps.groupby(pair_keys, as_index=False)[diagnostic_columns + eigenvalue_columns]
        .mean()
    )
    pair_summary = (
        per_repeat_pair.groupby(
            ["response", "left_tissue", "right_tissue"], as_index=False
        )[diagnostic_columns + eigenvalue_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    pair_summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in pair_summary.columns
    ]
    overlap_overall = (
        per_repeat_pair.groupby(["cv_repeat", "response"], as_index=False)[
            diagnostic_columns
        ]
        .mean()
        .groupby("response", as_index=False)[diagnostic_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    overlap_overall.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in overlap_overall.columns
    ]

    cover_fold = (
        metrics.loc[metrics["method"] == "COVER"]
        .groupby(["cv_repeat", "response", "test_fold"], as_index=False)
        ["selected_coupling"]
        .first()
    )
    coupling_counts = (
        cover_fold.groupby(["response", "selected_coupling"], as_index=False)
        .size()
        .rename(columns={"size": "folds_selected"})
    )

    config_paths = sorted(args.input_dir.glob("repeat_*/*/fold_*/classical_config.json"))
    reproduction_deltas = [
        float(json.loads(path.read_text())["hps_reproduction_max_mse_delta"])
        for path in config_paths
    ]
    audit = {
        "repeats": args.repeats,
        "folds_per_repeat": 5,
        "responses": 2,
        "outer_fold_jobs": expected,
        "methods": list(METHODS),
        "neural_files": len(neural_paths),
        "classical_files": len(classical_paths),
        "overlap_files": len(overlap_paths),
        "diagnostic_mean_baselines": list(MEAN_BASELINES[1:]),
        "maximum_hps_reproduction_mse_delta": max(reproduction_deltas),
    }

    args.input_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.input_dir / "all_fold_task_metrics.csv", index=False)
    per_tissue.to_csv(args.input_dir / "per_repeat_tissue_metrics.csv", index=False)
    per_repeat_response.to_csv(
        args.input_dir / "per_repeat_response_metrics.csv", index=False
    )
    per_repeat_overall.to_csv(
        args.input_dir / "per_repeat_overall_metrics.csv", index=False
    )
    response_summary.to_csv(args.input_dir / "response_mean_sd.csv", index=False)
    overall_summary.to_csv(args.input_dir / "overall_mean_sd.csv", index=False)
    baseline_response_summary.to_csv(
        args.input_dir / "mean_baseline_response_mean_sd.csv", index=False
    )
    baseline_overall_summary.to_csv(
        args.input_dir / "mean_baseline_overall_mean_sd.csv", index=False
    )
    (args.input_dir / "mean_baseline_audit.json").write_text(
        json.dumps(mean_baseline_audit, indent=2) + "\n"
    )
    paired_gains.to_csv(args.input_dir / "paired_gains.csv", index=False)
    win_counts.to_csv(args.input_dir / "repeat_win_counts.csv", index=False)
    runtime_summary.to_csv(args.input_dir / "runtime_mean_sd.csv", index=False)
    overlaps.to_csv(args.input_dir / "all_fold_pairwise_overlap.csv", index=False)
    per_repeat_pair.to_csv(args.input_dir / "per_repeat_pairwise_overlap.csv", index=False)
    pair_summary.to_csv(args.input_dir / "pairwise_overlap_mean_sd.csv", index=False)
    overlap_overall.to_csv(args.input_dir / "overlap_overall_mean_sd.csv", index=False)
    coupling_counts.to_csv(args.input_dir / "coupling_selection_counts.csv", index=False)
    (args.input_dir / "aggregation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    print("Response-specific mean (SD)")
    print(
        response_summary.pivot(
            index="method", columns="response", values="standardized_mean_sd"
        ).to_string()
    )
    print("\nEqual-response mean (SD)")
    print(overall_summary[["method", "standardized_mean_sd"]].to_string(index=False))
    print("\nPaired COVER gains")
    print(paired_gains.to_string(index=False))
    print("\nAudit")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
