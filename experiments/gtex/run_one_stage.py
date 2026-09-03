"""Run one fair one-stage GTEx fold with validation-selected stopping."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from data import TARGET_BRAIN_TISSUES, build_fold, donor_fold_map, load_tasks
from experiment import (
    COUPLINGS,
    ExperimentConfig,
    derive_seed,
    fit_model,
    initialize_model,
    overlap_rows,
    predict_by_task,
    prediction_rows,
    shared_model,
    task_mse,
)
from cover_mtl.simulations.training import NeuralFit, seed_everything


def fit_timed(
    model,
    fold,
    config: ExperimentConfig,
    *,
    coupling: float,
    seed: int,
    device: str,
    validation_task_indices: tuple[int, ...] | None = None,
) -> tuple[NeuralFit, float]:
    start = time.perf_counter()
    fit = fit_model(
        model,
        fold,
        config,
        coupling=coupling,
        seed=seed,
        device=device,
        validation_task_indices=validation_task_indices,
    )
    return fit, time.perf_counter() - start


def fit_shared_candidates(
    fold,
    config: ExperimentConfig,
    *,
    response: str,
    test_fold: int,
    device: str,
    validation_task_indices: tuple[int, ...] | None = None,
) -> tuple[NeuralFit, NeuralFit, float, float, list[dict[str, float]]]:
    initialization_seed = derive_seed(
        config.model_seed, response, test_fold, "shared", "initialization"
    )
    training_seed = derive_seed(
        config.model_seed, response, test_fold, "shared", "training"
    )
    seed_everything(initialization_seed)
    initial_hps = shared_model(config, len(fold.tissues), "none")
    initial_state = initial_hps.state_dict()

    hps, hps_seconds = fit_timed(
        initial_hps,
        fold,
        config,
        coupling=0.0,
        seed=training_seed,
        device=device,
        validation_task_indices=validation_task_indices,
    )
    hps_validation_all = task_mse(
        fold.validation_y, predict_by_task(hps.model, fold.validation_x, device)
    )
    selected_indices = (
        tuple(range(len(fold.tissues)))
        if validation_task_indices is None
        else validation_task_indices
    )
    hps_validation = hps_validation_all[list(selected_indices)]
    candidates: list[tuple[float, float, NeuralFit, float]] = [
        (float(hps_validation.mean()), 0.0, hps, hps_seconds)
    ]
    tuning: list[dict[str, float]] = [
        {
            "coupling": 0.0,
            "validation_mse": float(hps_validation.mean()),
            "validation_relative_mse": 1.0,
            "best_step": float(hps.best_step),
            "fit_seconds": hps_seconds,
        }
    ]

    for coupling in COUPLINGS:
        model = shared_model(config, len(fold.tissues), "cover")
        missing, unexpected = model.load_state_dict(initial_state, strict=False)
        if unexpected or missing != ["consensus"]:
            raise RuntimeError(
                f"Common initialization failed for coupling {coupling}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.initialize_consensus_from_heads()
        fit, elapsed = fit_timed(
            model,
            fold,
            config,
            coupling=coupling,
            seed=training_seed,
            device=device,
            validation_task_indices=validation_task_indices,
        )
        validation_all = task_mse(
            fold.validation_y, predict_by_task(fit.model, fold.validation_x, device)
        )
        validation = validation_all[list(selected_indices)]
        validation_mean = float(validation.mean())
        tuning.append(
            {
                "coupling": coupling,
                "validation_mse": validation_mean,
                "validation_relative_mse": float(
                    (validation / hps_validation.clip(min=1e-8)).mean()
                ),
                "best_step": float(fit.best_step),
                "fit_seconds": elapsed,
            }
        )
        candidates.append((validation_mean, coupling, fit, elapsed))

    selected = min(candidates, key=lambda item: (item[0], item[1]))
    return hps, selected[2], selected[1], selected[3], tuning


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("prepared/gtex_v8_brain_module137.parquet"),
    )
    parser.add_argument("--response", choices=("JAM2", "SH2D2A"), required=True)
    parser.add_argument("--test-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cv-repeat", type=int, default=None)
    parser.add_argument(
        "--target-validation",
        choices=("all", "brain11"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/one_stage"))
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
    validation_task_indices: tuple[int, ...] | None = None
    if args.target_validation == "brain11":
        tissue_to_index = {tissue: index for index, tissue in enumerate(fold.tissues)}
        missing_targets = sorted(set(TARGET_BRAIN_TISSUES) - set(tissue_to_index))
        if missing_targets:
            raise ValueError(f"Target brain tissues are missing: {missing_targets}")
        validation_task_indices = tuple(
            tissue_to_index[tissue] for tissue in TARGET_BRAIN_TISSUES
        )
    output_root = (
        args.output_dir
        if args.cv_repeat is None
        else args.output_dir / f"repeat_{repeat_index:02d}"
    )
    output = output_root / args.response / f"fold_{args.test_fold}"
    output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []

    zeros = tuple(values * 0.0 for values in fold.test_y)
    metrics.extend(prediction_rows("TissueMean", zeros, fold, 0.0, 0.0))
    global_mean = float(fold.response_mean.mean())
    global_prediction = tuple(
        values * 0.0
        + (global_mean - fold.response_mean[index]) / fold.response_scale[index]
        for index, values in enumerate(fold.test_y)
    )
    metrics.extend(
        prediction_rows("GlobalMean", global_prediction, fold, 0.0, 0.0)
    )

    for method in ("Pool", "STL", "MMoE"):
        initialization_seed = derive_seed(
            config.model_seed, args.response, args.test_fold, method, "initialization"
        )
        model = initialize_model(method, config, len(fold.tissues), initialization_seed)
        fit, elapsed = fit_timed(
            model,
            fold,
            config,
            coupling=0.0,
            seed=derive_seed(
                config.model_seed, args.response, args.test_fold, method, "training"
            ),
            device=args.device,
            validation_task_indices=validation_task_indices,
        )
        metrics.extend(
            prediction_rows(
                method,
                predict_by_task(fit.model, fold.test_x, args.device),
                fold,
                0.0,
                elapsed,
            )
        )

    hps, cover, coupling, cover_seconds, tuning = fit_shared_candidates(
        fold,
        config,
        response=args.response,
        test_fold=args.test_fold,
        device=args.device,
        validation_task_indices=validation_task_indices,
    )
    hps_seconds = float(tuning[0]["fit_seconds"])
    metrics.extend(
        prediction_rows(
            "HPS",
            predict_by_task(hps.model, fold.test_x, args.device),
            fold,
            0.0,
            hps_seconds,
        )
    )
    metrics.extend(
        prediction_rows(
            "COVER",
            predict_by_task(cover.model, fold.test_x, args.device),
            fold,
            coupling,
            cover_seconds,
        )
    )
    pd.DataFrame(metrics).assign(
        response=args.response, test_fold=args.test_fold, cv_repeat=repeat_index
    ).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(tuning).assign(
        response=args.response, test_fold=args.test_fold, cv_repeat=repeat_index
    ).to_csv(output / "cover_tuning.csv", index=False)
    pd.DataFrame(
        overlap_rows(cover.model, fold, args.device, selected_coupling=coupling)
    ).assign(
        response=args.response, test_fold=args.test_fold, cv_repeat=repeat_index
    ).to_csv(output / "overlap.csv", index=False)
    metadata = {
        "protocol": "fair one-stage training with validation-selected checkpoint",
        "config": asdict(config),
        "response": args.response,
        "test_fold": args.test_fold,
        "cv_repeat": repeat_index,
        "device": args.device,
        "selected_coupling": coupling,
        "tissues": list(fold.tissues),
        "target_validation": args.target_validation,
        "validation_tissues": (
            list(fold.tissues)
            if validation_task_indices is None
            else [fold.tissues[index] for index in validation_task_indices]
        ),
        "shared_initialization_seed": derive_seed(
            config.model_seed, args.response, args.test_fold, "shared", "initialization"
        ),
        "shared_training_seed": derive_seed(
            config.model_seed, args.response, args.test_fold, "shared", "training"
        ),
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = (
        pd.DataFrame(metrics)
        .groupby("method", as_index=False)["standardized_mse"]
        .mean()
        .sort_values("standardized_mse")
    )
    print(summary.to_string(index=False))
    print(f"COVER selected coupling: {coupling:g}")
    print(pd.DataFrame(tuning).to_string(index=False))


if __name__ == "__main__":
    main()
