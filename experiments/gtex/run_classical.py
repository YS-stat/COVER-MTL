"""Add ARMUL and FLARCC to a completed one-stage GTEx fold."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import build_fold, donor_fold_map, load_tasks
from experiment import (
    ExperimentConfig,
    derive_seed,
    fit_model,
    predict_by_task,
    prediction_rows,
    shared_model,
)
from cover_mtl.models import FixedHeadCommonPredictor
from cover_mtl.simulations.baselines.armul import tune_armul
from cover_mtl.simulations.baselines.flarcc import tune_flarcc
from cover_mtl.simulations.training import predict_decomposition, seed_everything


CLASSICAL_RATIOS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)


def fixed_features(model, x_by_task, y_by_task, device: str):
    representations, residuals = [], []
    for x_task, y_task in zip(x_by_task, y_by_task):
        common, representation = predict_decomposition(model, x_task, device=device)
        representations.append(representation)
        residuals.append(y_task - common.reshape(-1))
    return representations, residuals


def refit_common(
    hps,
    coefficients: np.ndarray,
    fold,
    config: ExperimentConfig,
    *,
    method: str,
    response: str,
    test_fold: int,
    device: str,
):
    model = FixedHeadCommonPredictor(
        hps.common_network,
        hps.representation_network,
        torch.as_tensor(coefficients, dtype=torch.float32),
    )
    return fit_model(
        model,
        fold,
        config,
        coupling=0.0,
        seed=derive_seed(
            config.model_seed, response, test_fold, method, "common_refit"
        ),
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--response", choices=("JAM2", "SH2D2A"), required=True)
    parser.add_argument("--test-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cv-repeat", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.cv_repeat is not None and args.cv_repeat < 0:
        raise ValueError("cv-repeat must be nonnegative.")
    repeat_index = 0 if args.cv_repeat is None else args.cv_repeat
    config = ExperimentConfig()
    if repeat_index > 0:
        config = replace(
            config,
            fold_seed=derive_seed(config.fold_seed, "gtex_cv_repeat", repeat_index),
            model_seed=derive_seed(config.model_seed, "gtex_cv_repeat", repeat_index),
        )
    tasks, predictors = load_tasks(args.data, args.response)
    mapping = donor_fold_map(tasks, config.fold_seed, config.folds)
    fold = build_fold(
        tasks,
        predictors,
        mapping,
        args.test_fold,
        folds=config.folds,
        screened_features=config.screened_features,
        principal_components=config.principal_components,
    )
    output_root = (
        args.output_dir
        if args.cv_repeat is None
        else args.output_dir / f"repeat_{repeat_index:02d}"
    )
    output = output_root / args.response / f"fold_{args.test_fold}"
    output.mkdir(parents=True, exist_ok=True)

    initialization_seed = derive_seed(
        config.model_seed, args.response, args.test_fold, "shared", "initialization"
    )
    training_seed = derive_seed(
        config.model_seed, args.response, args.test_fold, "shared", "training"
    )
    seed_everything(initialization_seed)
    hps_model = shared_model(config, len(fold.tissues), "none")
    hps = fit_model(
        hps_model,
        fold,
        config,
        coupling=0.0,
        seed=training_seed,
        device=args.device,
    )

    stored_path = output / "metrics.csv"
    if stored_path.exists():
        stored = pd.read_csv(stored_path)
        stored_hps = stored.loc[stored["method"] == "HPS"].sort_values("tissue")
        reproduced = pd.DataFrame(
            prediction_rows(
                "HPS",
                predict_by_task(hps.model, fold.test_x, args.device),
                fold,
                0.0,
                0.0,
            )
        ).sort_values("tissue")
        difference = np.max(
            np.abs(
                stored_hps["standardized_mse"].to_numpy()
                - reproduced["standardized_mse"].to_numpy()
            )
        )
        if difference > 1e-6:
            raise RuntimeError(f"HPS reproduction failed: maximum MSE delta={difference}")
    else:
        difference = float("nan")

    train_z, train_residual = fixed_features(
        hps.model, fold.train_x, fold.train_y, args.device
    )
    validation_z, validation_residual = fixed_features(
        hps.model, fold.validation_x, fold.validation_y, args.device
    )

    metrics: list[dict[str, object]] = []
    tuning: list[dict[str, object]] = []

    start = time.perf_counter()
    armul = tune_armul(
        train_z,
        train_residual,
        validation_z,
        validation_residual,
        lambda_ratios=CLASSICAL_RATIOS,
    )
    armul_refit = refit_common(
        hps.model,
        armul.fit.coefficients,
        fold,
        config,
        method="ARMUL",
        response=args.response,
        test_fold=args.test_fold,
        device=args.device,
    )
    armul_seconds = time.perf_counter() - start
    metrics.extend(
        prediction_rows(
            "ARMUL",
            predict_by_task(armul_refit.model, fold.test_x, args.device),
            fold,
            armul.selected_ratio,
            armul_seconds,
        )
    )
    tuning.extend({"method": "ARMUL", **row} for row in armul.tuning_rows)

    start = time.perf_counter()
    flarcc_candidates = []
    for range_power in (0.0, 1.0):
        path = tune_flarcc(
            train_z,
            train_residual,
            validation_z,
            validation_residual,
            lambda_ratios=CLASSICAL_RATIOS,
            range_power=range_power,
            include_task_intercepts=False,
        )
        flarcc_candidates.append(
            (path.validation_mse, -path.selected_ratio, range_power, path)
        )
        tuning.extend(
            {"method": "FLARCC", "range_power": range_power, **row}
            for row in path.tuning_rows
        )
    _, _, selected_range_power, flarcc = min(
        flarcc_candidates, key=lambda item: (item[0], item[1], item[2])
    )
    flarcc_refit = refit_common(
        hps.model,
        flarcc.fit.coefficients,
        fold,
        config,
        method="FLARCC",
        response=args.response,
        test_fold=args.test_fold,
        device=args.device,
    )
    flarcc_seconds = time.perf_counter() - start
    metrics.extend(
        prediction_rows(
            "FLARCC",
            predict_by_task(flarcc_refit.model, fold.test_x, args.device),
            fold,
            flarcc.selected_ratio,
            flarcc_seconds,
        )
    )

    pd.DataFrame(metrics).assign(
        response=args.response, test_fold=args.test_fold, cv_repeat=repeat_index
    ).to_csv(output / "classical_metrics.csv", index=False)
    pd.DataFrame(tuning).assign(
        response=args.response, test_fold=args.test_fold, cv_repeat=repeat_index
    ).to_csv(output / "classical_tuning.csv", index=False)
    metadata = {
        "response": args.response,
        "test_fold": args.test_fold,
        "cv_repeat": repeat_index,
        "hps_reproduction_max_mse_delta": difference,
        "armul_selected_ratio": armul.selected_ratio,
        "flarcc_selected_ratio": flarcc.selected_ratio,
        "flarcc_selected_range_power": selected_range_power,
    }
    (output / "classical_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        pd.DataFrame(metrics)
        .groupby("method")["standardized_mse"]
        .mean()
        .sort_values()
        .to_string()
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
