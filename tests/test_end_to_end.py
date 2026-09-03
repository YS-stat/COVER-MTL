"""Small end-to-end check of the formal DGP, tuning, and output pipeline."""

from __future__ import annotations

import pandas as pd

from cover_mtl.simulations.config import NetworkConfig, OptimizationConfig
from cover_mtl.simulations.dgp import (
    SpectralNeuralConfig,
    generate_spectral_neural_replicate,
)
from cover_mtl.simulations.runner import run_replicate


def test_formal_pipeline_writes_complete_outputs(tmp_path):
    config = SpectralNeuralConfig(
        scenario="both_random",
        num_tasks=4,
        num_profiles=2,
        input_dim=9,
        representation_dim=4,
        active_per_profile=2,
        train_size=20,
        validation_size=20,
        test_size=30,
        low_mode_count=4,
        subspace_rank=2,
        moment_sample_size=100,
        seed=19,
    )
    data = generate_spectral_neural_replicate(config)
    network = NetworkConfig(
        common_hidden=(6,), representation_hidden=(6,), representation_dim=4,
    )
    optimization = OptimizationConfig(
        steps=4, batch_size_per_task=8, evaluation_interval=1, patience_evaluations=4,
    )
    coupling_optimization = OptimizationConfig(
        steps=3, batch_size_per_task=8, evaluation_interval=1, patience_evaluations=3,
    )
    run_replicate(
        scenario=config.scenario,
        replicate=0,
        output_dir=tmp_path,
        network_name="test",
        methods=("HPS", "COVER"),
        couplings=(0.1,),
        device="cpu",
        prepared_data=data,
        network_config=network,
        optimization_config=optimization,
        coupling_optimization_config=coupling_optimization,
    )
    job_dir = tmp_path / config.scenario / "rep_000"
    metrics = pd.read_csv(job_dir / "metrics.csv")
    task_metrics = pd.read_csv(job_dir / "task_metrics.csv")
    tuning = pd.read_csv(job_dir / "tuning.csv")
    assert set(metrics["method"]) == {"HPS", "COVER"}
    assert len(task_metrics) == 2 * config.num_tasks
    assert set(tuning["coupling"]) == {0.0, 0.1}
    assert (job_dir / "config.json").is_file()
