"""Registry of paper-facing simulation blocks and their expected outputs."""

from __future__ import annotations

from dataclasses import dataclass


FULL_METHODS = (
    "HPS",
    "Pool",
    "STL",
    "MMoE",
    "COVER",
    "ARMUL",
    "FLARCC",
)


@dataclass(frozen=True)
class SimulationBlock:
    """Expected layout of one formal simulation block."""

    name: str
    result_directory: str
    legacy_result_directory: str
    settings: tuple[str, ...]
    replicates_per_setting: int
    methods: tuple[str, ...]
    seed_policy: str = "independent"


FORMAL_BLOCKS = (
    SimulationBlock(
        name="primary",
        result_directory="main",
        legacy_result_directory="main_final",
        settings=(
            "within_0p200/both_overlap_aligned",
            "within_0p300/both_overlap_aligned",
        ),
        replicates_per_setting=100,
        methods=FULL_METHODS,
    ),
    SimulationBlock(
        name="factorial_controls",
        result_directory="controls",
        legacy_result_directory="main_controls_final",
        settings=(
            "within_0p200/homogeneous",
            "within_0p200/covariate_only",
            "within_0p200/posterior_only",
        ),
        replicates_per_setting=100,
        methods=FULL_METHODS,
    ),
    SimulationBlock(
        name="average_moment_ablation",
        result_directory="average_moment",
        legacy_result_directory="average_moment_appendix",
        settings=("within_0p300/both_overlap_aligned",),
        replicates_per_setting=100,
        methods=("HPS", "Average-Moment", "COVER"),
    ),
    SimulationBlock(
        name="random_alignment",
        result_directory="random_alignment",
        legacy_result_directory="random_alignment_appendix",
        settings=("within_0p200/both_random",),
        replicates_per_setting=100,
        methods=FULL_METHODS,
    ),
    SimulationBlock(
        name="initialization_and_architecture",
        result_directory="sensitivity",
        legacy_result_directory="sensitivity_final",
        settings=("base", "d12", "d36", "narrow", "wide", "deep"),
        replicates_per_setting=5,
        methods=("HPS", "COVER"),
        seed_policy="fixed",
    ),
    SimulationBlock(
        name="large_task_count",
        result_directory="scaling",
        legacy_result_directory="scaling_final_pilot",
        settings=("tasks_0024", "tasks_0096", "tasks_0192"),
        replicates_per_setting=100,
        methods=FULL_METHODS,
    ),
    SimulationBlock(
        name="high_input_dimension",
        result_directory="scaling",
        legacy_result_directory="highp_nuisance_within_0p30",
        settings=("dimension_0050", "dimension_0100", "dimension_0200"),
        replicates_per_setting=100,
        methods=FULL_METHODS,
    ),
    SimulationBlock(
        name="mechanism",
        result_directory="mechanism",
        legacy_result_directory="mechanism_spectral_pilot",
        settings=(
            "covariate/value_0p001",
            "covariate/value_0p03",
            "covariate/value_0p1",
            "covariate/value_0p3",
            "covariate/value_1",
            "posterior/value_0",
            "posterior/value_0p25",
            "posterior/value_0p5",
            "posterior/value_0p75",
            "posterior/value_1",
        ),
        replicates_per_setting=100,
        methods=("HPS", "Average-Moment", "COVER"),
    ),
    SimulationBlock(
        name="overlap_gated_outlier",
        result_directory="outlier",
        legacy_result_directory="outlier_decoupling_final",
        settings=(
            "overlap_0p001",
            "overlap_0p003",
            "overlap_0p01",
            "overlap_0p03",
            "overlap_1",
        ),
        replicates_per_setting=100,
        methods=("COVER",),
    ),
)
