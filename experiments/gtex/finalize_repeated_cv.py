"""Aggregate repeated GTEx CV automatically after every fold is complete."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--maximum-hours", type=float, default=6.0)
    args = parser.parse_args()
    expected = args.repeats * 2 * 5
    deadline = time.monotonic() + args.maximum_hours * 3600.0
    while time.monotonic() < deadline:
        neural = len(list(args.input_dir.glob("repeat_*/*/fold_*/metrics.csv")))
        classical = len(
            list(args.input_dir.glob("repeat_*/*/fold_*/classical_metrics.csv"))
        )
        print(f"neural={neural}/{expected} classical={classical}/{expected}", flush=True)
        if neural == expected and classical == expected:
            script = Path(__file__).resolve().parent / "aggregate_repeated_cv.py"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input-dir",
                    str(args.input_dir),
                    "--repeats",
                    str(args.repeats),
                ],
                check=False,
            )
            raise SystemExit(completed.returncode)
        time.sleep(args.poll_seconds)
    raise TimeoutError("Repeated CV did not finish within the allowed time.")


if __name__ == "__main__":
    main()
