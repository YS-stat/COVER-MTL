"""Launch formal primary, control, or ablation replicates across fixed GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ..experiments.primary import SCENARIOS
from ..runner import DEFAULT_COUPLINGS, DEFAULT_METHODS


@dataclass(frozen=True)
class Job:
    scenario: str
    within_profile_scale: float
    replicate: int

    @property
    def setting(self) -> str:
        return f"within_{self.within_profile_scale:.3f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--scenarios", default="both_overlap_aligned")
    parser.add_argument("--within-scales", default="0.20,0.30")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--couplings", default=",".join(map(str, (0.0,) + DEFAULT_COUPLINGS)),
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    scenarios = tuple(value for value in args.scenarios.split(",") if value)
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    within_scales = tuple(
        float(value) for value in args.within_scales.split(",") if value
    )
    gpu_ids = tuple(int(value) for value in args.gpus.split(",") if value)
    slots = tuple((gpu, slot) for slot in range(args.slots_per_gpu) for gpu in gpu_ids)
    jobs = [
        Job(scenario, within_scale, replicate)
        for scenario in scenarios
        for within_scale in within_scales
        for replicate in range(args.replicates)
    ]
    jobs = [
        job
        for job in jobs
        if not (
            args.output_dir
            / job.setting
            / job.scenario
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
    failures: list[tuple[str, float, int, int]] = []

    while jobs or running:
        for slot in slots:
            if slot in running or not jobs:
                continue
            gpu, _ = slot
            job = jobs.pop(0)
            job_dir = (
                args.output_dir
                / job.setting
                / job.scenario
                / f"rep_{job.replicate:03d}"
            )
            job_dir.mkdir(parents=True, exist_ok=True)
            log_stream = (job_dir / "run.log").open("wb")
            command = [
                sys.executable,
                "-m",
                "cover_mtl.simulations.experiments.primary",
                "--scenario",
                job.scenario,
                "--replicate",
                str(job.replicate),
                "--output-dir",
                str(args.output_dir / job.setting),
                "--device",
                f"cuda:{gpu}",
                "--within-profile-scale",
                str(job.within_profile_scale),
                "--methods",
                args.methods,
                "--couplings",
                args.couplings,
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
            if return_code:
                failures.append(
                    (job.scenario, job.within_profile_scale, job.replicate, return_code)
                )
            del running[slot]
    if failures:
        raise RuntimeError(f"Primary simulation jobs failed: {failures}")


if __name__ == "__main__":
    main()
