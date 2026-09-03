"""Distribute final one-axis scaling jobs over fixed GPU slots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    axis: str
    value: int
    replicate: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--task-values", default="24,96,192")
    parser.add_argument("--dimension-values", default="50,100,200")
    parser.add_argument("--methods", default="HPS,Pool,STL,MMoE,COVER,ARMUL,FLARCC")
    parser.add_argument("--task-couplings", default="0.01,0.03,0.1,0.3,1,3,10,30")
    parser.add_argument(
        "--dimension-couplings", default="0.01,0.03,0.1,0.3,1,3,10,30,100"
    )
    parser.add_argument("--posterior-scale", type=float, default=1.0)
    parser.add_argument("--within-profile-scale", type=float, default=0.30)
    parser.add_argument("--weak-variance", type=float, default=0.001)
    parser.add_argument("--tasks-per-profile", type=int, default=0)
    parser.add_argument("--task-train-size", type=int, default=100)
    parser.add_argument("--dimension-train-size", type=int, default=100)
    parser.add_argument(
        "--posterior-geometry",
        choices=("spectral", "clustered", "projected"),
        default="clustered",
    )
    parser.add_argument(
        "--rotation-structure", choices=("full", "signal_block"), default="full",
    )
    parser.add_argument(
        "--common-coordinate-mode",
        choices=("auto", "representation", "separate"),
        default="auto",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value)
    slots = tuple((gpu, slot) for slot in range(args.slots_per_gpu) for gpu in gpu_ids)
    values = {
        "tasks": tuple(int(value) for value in args.task_values.split(",") if value),
        "dimension": tuple(
            int(value) for value in args.dimension_values.split(",") if value
        ),
    }
    jobs = [
        Job(axis, value, replicate)
        for axis, axis_values in values.items()
        for value in axis_values
        for replicate in range(args.replicates)
    ]
    jobs = [
        job
        for job in jobs
        if not (
            args.output_dir
            / f"{job.axis}_{job.value:04d}"
            / "both_overlap_aligned"
            / f"rep_{job.replicate:03d}"
            / "metrics.csv"
        ).exists()
    ]
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment["OMP_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["MKL_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads_per_job)
    running: dict[tuple[int, int], tuple[subprocess.Popen[bytes], Job, object]] = {}
    failures: list[tuple[str, int, int, int]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    while jobs or running:
        for slot in slots:
            if slot in running or not jobs:
                continue
            gpu, _ = slot
            job = jobs.pop(0)
            job_dir = (
                args.output_dir
                / f"{job.axis}_{job.value:04d}"
                / "both_overlap_aligned"
                / f"rep_{job.replicate:03d}"
            )
            job_dir.mkdir(parents=True, exist_ok=True)
            log_stream = (job_dir / "run.log").open("wb")
            command = [
                sys.executable,
                "-m",
                "cover_mtl.simulations.experiments.scaling",
                "--axis",
                job.axis,
                "--value",
                str(job.value),
                "--replicate",
                str(job.replicate),
                "--output-dir",
                str(args.output_dir),
                "--device",
                f"cuda:{gpu}",
                "--methods",
                args.methods,
                "--couplings",
                (
                    args.task_couplings
                    if job.axis == "tasks"
                    else args.dimension_couplings
                ),
                "--posterior-scale",
                str(args.posterior_scale),
                "--within-profile-scale",
                str(args.within_profile_scale),
                "--weak-variance",
                str(args.weak_variance),
                "--tasks-per-profile",
                str(args.tasks_per_profile),
                "--task-train-size",
                str(args.task_train_size),
                "--dimension-train-size",
                str(args.dimension_train_size),
                "--posterior-geometry",
                args.posterior_geometry,
                "--rotation-structure",
                (args.rotation_structure if job.axis == "tasks" else "signal_block"),
                "--common-coordinate-mode",
                (
                    args.common_coordinate_mode
                    if job.axis == "tasks"
                    else "representation"
                ),
            ]
            process = subprocess.Popen(
                command, stdout=log_stream, stderr=subprocess.STDOUT, env=environment,
            )
            running[slot] = (process, job, log_stream)
        time.sleep(args.poll_seconds)
        for slot, (process, job, log_stream) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            if return_code != 0:
                failures.append((job.axis, job.value, job.replicate, return_code))
            del running[slot]

    if failures:
        raise RuntimeError(f"Final scaling jobs failed: {failures}")


if __name__ == "__main__":
    main()
