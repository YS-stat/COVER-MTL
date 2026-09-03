"""Tests for stable experiment seed derivation."""

from cover_mtl.simulations.randomness import derive_seed


def test_seed_derivation_is_stable_and_label_specific():
    first = derive_seed(10, "scenario", 3, "cover")
    second = derive_seed(10, "scenario", 3, "cover")
    different = derive_seed(10, "scenario", 3, "hps")
    assert first == second
    assert first != different
    assert 0 <= first < 2 ** 31 - 1
