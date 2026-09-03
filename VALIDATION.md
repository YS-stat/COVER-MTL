# Implementation Validation

The release was checked against the scripts that generated the formal revision
results. The checks cover model provenance, objective scaling, data generation,
tuning, evaluation, and output completeness.

## Model and objective checks

- The formal implementation uses distinct networks for the common function
  `g` and representation `z`.
- Task heads are projected onto the centered subspace after every optimizer
  update, so their sum is zero to numerical precision.
- Balanced mini-batches give equal task weight in the empirical prediction
  loss.
- The auxiliary COVER coefficient is `2 * lambda / (T * (T - 1))`. Profiling
  the pairwise consensus variables recovers the stated overlap penalty.
- Unit tests verify the profiling identity for positive-definite,
  noncommuting, and singular positive-semidefinite second moments.
- The Average-Moment ablation uses the pairwise arithmetic second moment with
  its corresponding objective coefficient.

## DGP, tuning, and metric checks

- Homogeneous, covariate-only, posterior-only, and double-heterogeneity DGPs
  switch only their intended population components.
- The true task coefficients are centered, and the stored common and deviation
  functions reconstruct every conditional mean.
- Training, validation, and test data are generated independently. Validation
  responses select tuning parameters and checkpoints; test responses are used
  only for final evaluation.
- Excess MSE is computed against the conditional mean. Prediction MSE is
  computed against the noisy response. Both are averaged equally across tasks.
- Summary metrics were independently reconstructed from all task-level rows.

## Automated checks

- 26 unit and end-to-end tests pass.
- The formal output audit covers 9 experiment blocks, 34 settings, and 2,830
  completed replicates. It found no missing replicates, unexpected methods,
  nonfinite key metrics, task-summary mismatches, seed collisions, or coupling
  selections outside their validation grids.
- The fixed-representation Monte Carlo verification contains 155 rows over five
  designs. Its maximum relative discrepancy from the theoretical identity is
  0.004319.

The machine-readable audit is stored in `formal_simulation_audit.json`.

## Reporting conventions

- ARMUL uses the prespecified 5,000-iteration computational budget.
- `workflow_seconds` includes HPS initialization for methods that depend on the
  HPS representation.
- `parameter_count` denotes parameters needed by the fitted prediction rule.
  Pairwise consensus variables are reported separately as optimization
  auxiliaries and are not counted as predictive parameters.
- The mechanism axis labeled `posterior` changes the between-profile posterior
  strength while holding the within-profile scale at 0.30. The covariate axis
  changes an overlap level, so larger values correspond to less covariate
  heterogeneity.
