# Formal Simulation Protocol

This document records the frozen settings used by the paper-facing simulation
suite. It excludes exploratory designs, calibration runs, and real-data code.

## Common model and evaluation

All neural experiments fit a common network `g` and a separate representation
network `z`. HPS, Average-Moment, and COVER use the same architecture and
centered task heads. COVER is initialized from the HPS fit and optimized with
the exact pairwise auxiliary-consensus objective.

Each replicate generates independent training, validation, and test splits.
Validation responses select the coupling and early-stopping checkpoint. Test
responses are read only after selection. All methods in a replicate use the
same three splits. The primary metric is task-balanced excess MSE against the
known conditional mean; prediction MSE, worst-task excess MSE, common-function
error, task-deviation error, time, memory, and model size are also stored.

The neural optimizer uses AdamW with learning rate `1e-3`, gradient clipping at
`5`, and balanced mini-batches. The initial neural fit uses at most 2,500 steps;
coupled fine-tuning uses at most 1,000 steps. ARMUL is intentionally run with a
5,000-iteration computational budget. All tuning grids include the uncoupled
fit. Random seeds are deterministic functions of the block, setting, method,
and replicate.

## Primary and factorial experiments

The primary DGP uses 48 tasks, 6 covariate profiles, 24 input and representation
coordinates, 100 training observations per task, 200 validation observations,
and 3,000 test observations. The common network has width 32; the representation
network has width 48 and a 24-dimensional output. Both use ReLU activations.
The task-specific signal uses a tanh population representation. The two main
settings use within-profile posterior scales 0.20 and 0.30.

```bash
python -m cover_mtl.simulations.launchers.primary \
  --output-dir results/main \
  --replicates 100 \
  --scenarios both_overlap_aligned \
  --within-scales 0.20,0.30
```

The factorial controls hold the same architecture and use homogeneous,
covariate-only, and posterior-only settings.

```bash
python -m cover_mtl.simulations.launchers.primary \
  --output-dir results/controls \
  --replicates 100 \
  --scenarios homogeneous,covariate_only,posterior_only \
  --within-scales 0.20
```

The random-alignment robustness setting and Average-Moment ablation are:

```bash
python -m cover_mtl.simulations.launchers.primary \
  --output-dir results/random_alignment \
  --replicates 100 \
  --scenarios both_random \
  --within-scales 0.20

python -m cover_mtl.simulations.launchers.primary \
  --output-dir results/average_moment \
  --replicates 100 \
  --scenarios both_overlap_aligned \
  --within-scales 0.30 \
  --methods HPS,Average-Moment,COVER
```

## Sensitivity and scaling

Initialization and architecture sensitivity uses one frozen data set and five
independent model initializations for each of six architectures.

```bash
python -m cover_mtl.simulations.launchers.sensitivity \
  --output-dir results/sensitivity \
  --initializations 5
```

The scaling launcher reproduces task counts 24, 96, and 192 and input dimensions
50, 100, and 200. The high-input-dimensional settings randomly rotate the predictive
24-dimensional block and add independent nuisance coordinates.

```bash
python -m cover_mtl.simulations.launchers.scaling \
  --output-dir results/scaling \
  --replicates 100
```

## Mechanism and overlap-gated outlier experiments

The mechanism experiment changes one population axis at a time. The weak
variance, an overlap level that increases as covariate heterogeneity decreases,
uses values `0.001, 0.03, 0.1, 0.3, 1`. The between-profile posterior strength
uses values `0, 0.25, 0.5, 0.75, 1`, while the within-profile posterior scale is
held at `0.30`. Each point has 100 replicates and compares HPS, Average-Moment,
and COVER.

```bash
python -m cover_mtl.simulations.launchers.mechanism \
  --output-dir results/mechanism \
  --replicates 100 \
  --slots-per-gpu 2
```

The outlier diagnostic varies the overlap of one posterior-outlying task and
reports COVER alone, as intended by the paper's scope statement.

```bash
python -m cover_mtl.simulations.launchers.outlier \
  --output-dir results/outlier \
  --replicates 100
```

## Theory verification and audit

```bash
python -m cover_mtl.simulations.experiments.theory \
  --output results/theory_verification/fixed_representation.csv \
  --draws 2000 \
  --lambda-count 31

python -m cover_mtl.simulations.results.audit \
  --results-root results
```

The audit verifies expected replicate IDs, method and task rows, finite key
metrics, coupling-grid membership, deterministic seed policy, configuration
consistency, empty run logs, and the fixed-representation identity.
