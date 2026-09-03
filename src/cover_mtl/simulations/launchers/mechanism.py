"""Launch covariate- and posterior-strength experiments on multiple GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ..runner import DEFAULT_COUPLINGS

AXIS_VALUES = {
    "covariate": (1.0, 0.3, 0.1, 0.03, 0.001),
    "posterior": (0.0, 0.25, 0.5, 0.75, 1.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", required=True, type=int)
    parser.add_argument(
        "--couplings", default=",".join(map(str, (0.0,) + DEFAULT_COUPLINGS))
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--cpu-threads-per-job", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU identifier is required.")
    if args.slots_per_gpu < 1:
        raise ValueError("slots-per-gpu must be positive.")
    if args.cpu_threads_per_job < 1:
        raise ValueError("cpu-threads-per-job must be positive.")
    jobs = []
    for axis, values in AXIS_VALUES.items():
        for value in values:
            tag = f"{value:.4g}".replace(".", "p")
            for replicate in range(args.replicates):
                output = (
                    args.output_dir
                    / axis
                    / f"value_{tag}"
                    / f"rep_{replicate:03d}"
                    / "metrics.csv"
                )
                if not output.exists():
                    jobs.append((axis, value, replicate))
    slots = [(gpu, slot) for slot in range(args.slots_per_gpu) for gpu in gpu_ids]
    running: dict[
        tuple[str, int], tuple[subprocess.Popen[bytes], tuple[str, float, int], object],
    ] = {}
    failures: list[tuple[str, float, int, int]] = []
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment["OMP_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["MKL_NUM_THREADS"] = str(args.cpu_threads_per_job)
    environment["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads_per_job)
    while jobs or running:
        for slot_key in slots:
            if slot_key in running or not jobs:
                continue
            gpu, _ = slot_key
            axis, value, replicate = jobs.pop(0)
            tag = f"{value:.4g}".replace(".", "p")
            job_dir = args.output_dir / axis / f"value_{tag}" / f"rep_{replicate:03d}"
            job_dir.mkdir(parents=True, exist_ok=True)
            log_stream = (job_dir / "run.log").open("wb")
            command = [
                sys.executable,
                "-m",
                "cover_mtl.simulations.experiments.mechanism",
                "--axis",
                axis,
                "--value",
                str(value),
                "--replicate",
                str(replicate),
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
            running[slot_key] = (process, (axis, value, replicate), log_stream)
        time.sleep(args.poll_seconds)
        for slot_key, (process, job, log_stream) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_stream.close()
            if return_code != 0:
                failures.append((*job, return_code))
            del running[slot_key]
    if failures:
        raise RuntimeError(f"Mechanism experiment jobs failed: {failures}")


if __name__ == "__main__":
    main()
