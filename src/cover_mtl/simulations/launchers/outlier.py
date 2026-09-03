"""Launch the COVER-only outlier-decoupling experiment on fixed GPU slots."""

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
    overlap: float
    replicate: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--overlap-values", default="0.001,0.003,0.01,0.03,1")
    parser.add_argument("--couplings", default="0,0.01,0.03,0.1,0.3,1,3,10,30")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    overlaps = tuple(float(value) for value in args.overlap_values.split(",") if value)
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value)
    slots = tuple((gpu, slot) for slot in range(args.slots_per_gpu) for gpu in gpu_ids)
    jobs = []
    for overlap in overlaps:
        tag = f"{overlap:.4g}".replace(".", "p")
        for replicate in range(args.replicates):
            output = (
                args.output_dir
                / f"overlap_{tag}"
                / f"rep_{replicate:03d}"
                / "metrics.csv"
            )
            if not output.exists():
                jobs.append(Job(overlap, replicate))

    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment["OMP_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["MKL_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads_per_job)
    running: dict[tuple[int, int], tuple[subprocess.Popen[bytes], Job, object]] = {}
    failures: list[tuple[float, int, int]] = []

    while jobs or running:
        for slot in slots:
            if slot in running or not jobs:
                continue
            gpu, _ = slot
            job = jobs.pop(0)
            tag = f"{job.overlap:.4g}".replace(".", "p")
            job_dir = args.output_dir / f"overlap_{tag}" / f"rep_{job.replicate:03d}"
            job_dir.mkdir(parents=True, exist_ok=True)
            log_stream = (job_dir / "run.log").open("wb")
            command = [
                sys.executable,
                "-m",
                "cover_mtl.simulations.experiments.outlier",
                "--overlap",
                str(job.overlap),
                "--replicate",
                str(job.replicate),
                "--output-dir",
                str(args.output_dir),
                "--couplings",
                args.couplings,
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
                failures.append((job.overlap, job.replicate, return_code))
            del running[slot]
    if failures:
        raise RuntimeError(f"Outlier-decoupling jobs failed: {failures}")


if __name__ == "__main__":
    main()
