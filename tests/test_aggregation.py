"""Checks for paper-facing aggregation corrections and setting labels."""

from __future__ import annotations

import json

import pandas as pd

from cover_mtl.simulations.results.aggregate import (
    _add_workflow_seconds,
    _read_with_setting,
)


def test_legacy_counts_and_nested_main_setting_are_normalized(tmp_path):
    job_dir = tmp_path / "within_0p200" / "posterior_only" / "rep_000"
    job_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "method": ["HPS", "COVER", "ARMUL", "FLARCC"],
            "selected_coupling": [0.0, 0.3, 0.1, 0.1],
            "parameter_count": [100, 220, 90, 90],
            "fit_seconds": [2.0, 5.0, 1.0, 1.5],
            "replicate": [0, 0, 0, 0],
        }
    ).to_csv(job_dir / "metrics.csv", index=False)
    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "dgp": {"num_tasks": 4, "input_dim": 9},
                "network": {"representation_dim": 3},
            }
        ),
        encoding="utf-8",
    )
    frame = _read_with_setting(job_dir / "metrics.csv", tmp_path, "main")
    assert set(frame["setting"]) == {"within_0p200/posterior_only"}
    counts = frame.set_index("method")["parameter_count"]
    assert counts["HPS"] == 100
    assert counts["COVER"] == 220 - 6 * 3
    assert counts["ARMUL"] == 90 + 4 * 3
    assert counts["FLARCC"] == 90 + 4 * 3
    workflows = _add_workflow_seconds(frame).set_index("method")["workflow_seconds"]
    assert workflows["HPS"] == 2.0
    assert workflows["COVER"] == 7.0
    assert workflows["ARMUL"] == 3.0
    assert workflows["FLARCC"] == 3.5


def test_current_schema_does_not_recorrect_prediction_counts(tmp_path):
    job_dir = tmp_path / "tasks_0024" / "both_overlap_aligned" / "rep_000"
    job_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "method": ["COVER", "ARMUL"],
            "selected_coupling": [0.3, 0.1],
            "parameter_count": [102, 102],
            "fit_seconds": [1.0, 1.0],
            "replicate": [0, 0],
        }
    ).to_csv(job_dir / "metrics.csv", index=False)
    (job_dir / "config.json").write_text(
        json.dumps(
            {
                "output_schema_version": 2,
                "dgp": {"num_tasks": 4, "input_dim": 9},
                "network": {"representation_dim": 3},
            }
        ),
        encoding="utf-8",
    )
    frame = _read_with_setting(job_dir / "metrics.csv", tmp_path, "scaling")
    assert frame["parameter_count"].tolist() == [102, 102]
