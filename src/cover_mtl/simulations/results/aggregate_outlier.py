"""Aggregate the COVER-only outlier-decoupling experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _standard_error(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=np.float64)
    if array.size <= 1:
        return float("nan")
    return float(array.std(ddof=1) / np.sqrt(array.size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-replicates", type=int, default=100)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("overlap_*/rep_*/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics found below {args.input_dir}.")
    metrics = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"replicate", "outlier_task"}
    ]
    rows = []
    for overlap, frame in metrics.groupby("overlap"):
        row: dict[str, object] = {
            "overlap": overlap,
            "replicates": int(frame["replicate"].nunique()),
        }
        for column in numeric:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_se"] = _standard_error(frame[column])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("overlap")
    complete = bool(
        summary.shape[0] == 5
        and (summary["replicates"] == args.expected_replicates).all()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "all_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    with (args.output_dir / "aggregation_audit.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "metric_files": len(files),
                "expected_replicates": args.expected_replicates,
                "complete": complete,
            },
            stream,
            indent=2,
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
