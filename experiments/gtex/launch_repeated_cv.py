"""Launch the frozen 20-by-5 donor-level GTEx analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_command(command: list[str], log_path: Path, environment: dict[str, str]) -> int:
    with log_path.open("a") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return completed.returncode


def run_fold(
    root: Path,
    data: Path,
    output_dir: Path,
    response: str,
    repeat: int,
    fold: int,
    device: str,
    threads: int,
) -> tuple[int, str, int, int]:
    destination = output_dir / f"repeat_{repeat:02d}" / response / f"fold_{fold}"
    log_path = output_dir / "logs" / f"repeat_{repeat:02d}_{response}_fold_{fold}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    common = [
        "--data",
        str(data),
        "--response",
        response,
        "--test-fold",
        str(fold),
        "--cv-repeat",
        str(repeat),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if not (destination / "metrics.csv").exists():
        returncode = run_command(
            [sys.executable, str(root / "run_one_stage.py"), *common],
            log_path,
            environment,
        )
        if returncode != 0:
            return repeat, response, fold, returncode
    if not (destination / "classical_metrics.csv").exists():
        returncode = run_command(
            [sys.executable, str(root / "run_classical.py"), *common],
            log_path,
            environment,
        )
        if returncode != 0:
            return repeat, response, fold, returncode
    return repeat, response, fold, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads-per-job", type=int, default=2)
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    args = parser.parse_args()
    if min(args.repeats, args.workers, args.threads_per_job) <= 0:
        raise ValueError("Repeats, workers, and thread counts must be positive.")
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices:
        raise ValueError("At least one device is required.")
    root = Path(__file__).resolve().parent
    jobs = [
        (repeat, response, fold)
        for repeat in range(args.repeats)
        for response in ("JAM2", "SH2D2A")
        for fold in range(5)
    ]
    failures: list[tuple[int, str, int, int]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_fold,
                root,
                args.data,
                args.output_dir,
                response,
                repeat,
                fold,
                devices[index % len(devices)],
                args.threads_per_job,
            ): (repeat, response, fold)
            for index, (repeat, response, fold) in enumerate(jobs)
        }
        for future in as_completed(futures):
            repeat, response, fold, returncode = future.result()
            print(
                f"repeat {repeat:02d} {response} fold {fold}: "
                f"return code {returncode}",
                flush=True,
            )
            if returncode != 0:
                failures.append((repeat, response, fold, returncode))
    if failures:
        raise SystemExit(f"Repeated-CV jobs failed: {failures}")


if __name__ == "__main__":
    main()
