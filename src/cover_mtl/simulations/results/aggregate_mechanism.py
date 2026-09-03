"""Aggregate covariate- and posterior-strength experiment tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _standard_error(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    return (
        float(array.std(ddof=1) / np.sqrt(array.size))
        if array.size > 1
        else float("nan")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    metric_files = sorted(args.input_dir.glob("**/metrics.csv"))
    tuning_files = sorted(args.input_dir.glob("**/tuning.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No mechanism results found below {args.input_dir}.")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    tuning = pd.concat([pd.read_csv(path) for path in tuning_files], ignore_index=True)
    rows = []
    for (axis, value, method), frame in metrics.groupby(["axis", "value", "method"]):
        rows.append(
            {
                "axis": axis,
                "value": value,
                "method": method,
                "replicates": frame["replicate"].nunique(),
                "excess_mse_mean": float(frame["excess_mse"].mean()),
                "excess_mse_std": float(frame["excess_mse"].std(ddof=1)),
                "excess_mse_se": _standard_error(frame["excess_mse"]),
                "common_mse_mean": float(frame["common_mse"].mean()),
                "deviation_mse_mean": float(frame["deviation_mse"].mean()),
                "deviation_mse_std": float(frame["deviation_mse"].std(ddof=1)),
                "selected_coupling_median": float(frame["selected_coupling"].median()),
                "normalized_overlap_strength_mean": float(
                    frame["normalized_overlap_strength"].mean()
                ),
                "overlap_energy_mean": float(frame["overlap_energy"].mean()),
                "supported_contrast_fraction_mean": float(
                    frame["supported_contrast_fraction"].mean()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["axis", "value", "method"])
    paired_rows = []
    for (axis, value), frame in metrics.groupby(["axis", "value"]):
        indexed = {
            method: frame[frame["method"] == method].set_index("replicate")
            for method in ("HPS", "Average-Moment", "COVER")
        }
        for reference in ("HPS", "Average-Moment"):
            shared = indexed["COVER"].index.intersection(indexed[reference].index)
            difference = (
                indexed["COVER"].loc[shared, "excess_mse"]
                - indexed[reference].loc[shared, "excess_mse"]
            )
            paired_rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "reference": reference,
                    "replicates": len(shared),
                    "cover_minus_reference_mean": float(difference.mean()),
                    "cover_minus_reference_std": float(difference.std(ddof=1)),
                    "cover_minus_reference_se": _standard_error(difference),
                }
            )
    paired = pd.DataFrame(paired_rows).sort_values(["axis", "value"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "all_metrics.csv", index=False)
    tuning.to_csv(args.output_dir / "all_tuning.csv", index=False)
    summary.to_csv(args.output_dir / "mechanism_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_cover_gains.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
