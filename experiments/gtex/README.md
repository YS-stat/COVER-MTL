# GTEx central-nervous-system experiment for COVER-MTL

This directory contains the frozen GTEx v8 experiment used to evaluate COVER-MTL.
Only the final seven-method, donor-disjoint repeated cross-validation protocol is
retained.

## Scientific setting

- Tasks: 11 central-nervous-system tissues, including cervical spinal cord.
- Responses: `JAM2` and `SH2D2A`.
- Candidate predictors: genes in MODULE 137, excluding both response genes.
- Outcome scale: `log2(TPM + 1)`.
- Splitting unit: donor. All samples from the same donor remain in the same fold.
- Evaluation: task-balanced standardized and raw prediction mean squared error.
- Validation: feature screening, scaling, principal components, stopping time, and
  method-specific regularization are selected without using test donors.

The final analysis uses 20 independently seeded repetitions of donor-level five-fold
cross-validation. Each repetition produces one out-of-fold estimate per response and
method. Reported uncertainty is the empirical standard deviation across the 20 complete
cross-validation estimates, not across the 100 overlapping folds.

## Compared methods

The frozen comparison includes `COVER`, `ARMUL`, `FLARCC`, `HPS`, `Pool`, `STL`, and
`MMoE`. Tissue-specific and global training means are retained as diagnostic baselines
inside the fold-level neural output but are not included in the seven-method ranking.

## Directory layout

- `data_manifest.txt`: public source URLs for the 11 tissue files and MODULE 137 list.
- `download_data.py`: downloads the files listed in the public data manifest.
- `prepare_data.py`: constructs the analysis-ready central-nervous-system data.
- `data.py`: donor splitting and train-only preprocessing.
- `experiment.py`: frozen configuration and shared model utilities.
- `run_one_stage.py`: neural methods and COVER tuning for one response-fold job.
- `run_classical.py`: ARMUL and FLARCC for the same prepared response-fold job.
- `launch_repeated_cv.py`: launches all 20 x 2 x 5 jobs.
- `aggregate_repeated_cv.py`: verifies and aggregates the completed experiment.
- `finalize_repeated_cv.py`: waits for all jobs before aggregation.
- `prepared/gtex_v8_brain_module137.parquet`: generated analysis-ready data.
- `results/repeated_cv_v2/`: generated fold-level and aggregated results.
- `frozen_results/`: the machine-readable aggregates used in the manuscript.

## Reproduction

From the repository root:

```bash
python experiments/gtex/download_data.py

python experiments/gtex/prepare_data.py \
  --raw-dir experiments/gtex/raw \
  --output-dir experiments/gtex/prepared

python experiments/gtex/launch_repeated_cv.py \
  --data experiments/gtex/prepared/gtex_v8_brain_module137.parquet \
  --output-dir experiments/gtex/results/repeated_cv_v2 \
  --repeats 20 --workers 12 --threads-per-job 2 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3

python experiments/gtex/aggregate_repeated_cv.py \
  --input-dir experiments/gtex/results/repeated_cv_v2 \
  --repeats 20
```

## Frozen prediction results

Task-balanced standardized prediction MSE, reported as mean (standard deviation):

| Response | COVER | ARMUL | FLARCC | HPS | Pool | STL | MMoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| JAM2 | 0.2808 (0.0099) | 0.2806 (0.0088) | 0.2858 (0.0104) | 0.2845 (0.0099) | 0.3175 (0.0092) | 0.3220 (0.0098) | 0.3139 (0.0170) |
| SH2D2A | 0.6802 (0.0429) | 0.6831 (0.0435) | 0.6887 (0.0447) | 0.6918 (0.0440) | 0.6786 (0.0400) | 0.7845 (0.0524) | 0.7503 (0.0565) |
| Equal-response average | **0.4805 (0.0203)** | 0.4819 (0.0198) | 0.4872 (0.0211) | 0.4882 (0.0207) | 0.4981 (0.0212) | 0.5532 (0.0275) | 0.5321 (0.0289) |

Across the 22 response-tissue combinations, COVER has the largest number of
first-place finishes (7) and ties ARMUL for the largest number of top-three finishes
(19). It outperforms HPS in 17 of the 22 combinations.

## Mean-baseline audit

The GTEx result is not explained by shrinkage toward an intercept-only predictor.
COVER beats both diagnostic mean baselines in every response-tissue combination and
in every complete repeated-CV estimate.

| Method | Overall standardized MSE | Overall raw MSE |
|---|---:|---:|
| COVER | 0.4805 (0.0203) | 0.0631 (0.0013) |
| TissueMean | 1.1401 (0.0331) | 0.1901 (0.0012) |
| GlobalMean | 1.4736 (0.0323) | 0.2439 (0.0005) |

Relative to TissueMean and GlobalMean, COVER reduces standardized prediction error by
57.9% and 67.4%, respectively.

## Integrity checks

The aggregation audit requires 200 neural files, 200 classical files, and 200 overlap
files. Reproducing HPS independently in the neural and classical pipelines yields a
maximum MSE discrepancy of `8.88e-16`. COVER records the selected coupling and the
learned pairwise overlap diagnostics for every response-fold job.
The reported heatmap uses the mean generalized eigenvalue of the overlap
matrix relative to the pairwise average second-moment matrix. This diagnostic
is invariant to an invertible change of coordinates in the learned
representation.
