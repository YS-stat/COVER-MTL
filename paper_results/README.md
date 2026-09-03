# COVER-MTL Paper Results

This directory contains manuscript-facing tables and figures derived from the
frozen simulation and GTEx results.  The build script reads the formal result
directories without modifying them.

Run from the project root with:

```bash
python paper_results/build_paper_results.py
```

Outputs are organized as follows:

- `tables/main`: LaTeX and CSV summaries intended for the main paper;
- `tables/additional`: LaTeX summaries intended for Additional Experiments;
- `figures/main`: final PDF versions of main-paper figures;
- `figures/additional`: final PDF versions of supplementary figures;
- `data`: machine-readable aggregates used by the tables and figures;
- `provenance.json`: frozen input directories and exclusions.

Simulation performance figures use task-balanced excess mean squared error.
Curves show Monte Carlo means with 95% confidence intervals.  GTEx tables use
20 repeated five-fold cross-validation partitions and report means with
between-repeat standard deviations.
