"""Stable random-seed derivation for paired and parallel experiments."""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, *labels: object) -> int:
    """Derive a reproducible 31-bit seed independent of process hash state."""
    payload = "|".join([str(int(base_seed)), *(str(label) for label in labels)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2 ** 31 - 1)
