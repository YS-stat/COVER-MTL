"""Launch initialization and architecture sensitivity jobs on GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..experiments.sensitivity import NETWORKS


@dataclass(frozen=True)
class Job:
    variant: str
    initialization: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--initializations", type=int, default=5)
    parser.add_argument("--variants", default=",".join(NETWORKS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    variants = tuple(value for value in args.variants.split(",") if value)
    unknown = sorted(set(variants) - set(NETWORKS))
    if unknown:
        raise ValueError(f"Unknown sensitivity variants: {unknown}")
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value)
    slots = tuple((gpu, slot) for slot in range(args.slots_per_gpu) for gpu in gpu_ids)
    jobs = [
        Job(variant, initialization)
        for variant in variants
        for initialization in range(args.initializations)
    ]
    jobs = [
        job
        for job in jobs
        if not (
            args.output_dir
            / job.variant
            / "both_overlap_aligned"
            / f"rep_{job.initialization:03d}"
            / "metrics.csv"
        ).exists()
    ]
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment["OMP_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["MKL_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads_per_job)
    running: dict[tuple[int, int], tuple[subprocess.Popen[bytes], Job, object]] = {}
    failures: list[tuple[str, int, int]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    while jobs or running:
        for slot in slots:
            if slot in running or not jobs:
                continue
            gpu, _ = slot
            job = jobs.pop(0)
            job_dir = (
                args.output_dir
                / job.variant
                / "both_overlap_aligned"
                / f"rep_{job.initialization:03d}"
            )
            job_dir.mkdir(parents=True, exist_ok=True)
            log_stream = (job_dir / "run.log").open("wb")
            command = [
                sys.executable,
                "-m",
                "cover_mtl.simulations.experiments.sensitivity",
                "--variant",
                job.variant,
                "--initialization",
                str(job.initialization),
                "--output-dir",
                str(args.output_dir),
                "--device",
                f"cuda:{gpu}",
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
                failures.append((job.variant, job.initialization, return_code))
            del running[slot]
    if failures:
        raise RuntimeError(f"Sensitivity jobs failed: {failures}")


if __name__ == "__main__":
    main()
