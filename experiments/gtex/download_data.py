"""Download the public files used by the COVER-MTL GTEx experiment."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def download(url: str, destination: Path) -> None:
    """Download one URL atomically, leaving an existing file unchanged."""
    if destination.exists():
        print(f"exists: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "COVER-MTL/0.1"})
    print(f"download: {url}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data_manifest.txt")
    parser.add_argument("--output-dir", type=Path, default=root / "raw")
    args = parser.parse_args()

    urls = [
        line.strip()
        for line in args.manifest.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not urls:
        raise ValueError(f"No URLs were found in {args.manifest}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        filename = Path(urlparse(url).path).name
        if not filename:
            raise ValueError(f"Cannot determine a filename from {url!r}.")
        download(url, args.output_dir / filename)


if __name__ == "__main__":
    main()
