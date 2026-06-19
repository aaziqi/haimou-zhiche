# MP-CycleGAN Paper Materials

This directory contains the manuscript-side materials for the paper:

`MP-CycleGAN: Multi-Perceptual Constraints for Unpaired Underwater Image Enhancement`

## Included materials

- `spic_manuscript.tex`
  - final manuscript source used for the SPIC submission
- `highlights.txt`
  - final submission highlights
- `figures_used/`
  - figure assets used in the manuscript
- `revision_stats_from_csv.py`
  - script for paired delta analysis
- `revision_stats_euvp_250_vs_250.json`
  - paired statistics for EUVP
- `revision_stats_uieb_250_vs_250.json`
  - paired statistics for UIEB
- `build_cut_counterexample_figure.py`
  - script for the reviewer-oriented CUT/FastCUT counterexample figure
- `run_to250_continuation.ps1`, `run_to250_eval.ps1`
  - helper scripts used during the controlled continuation/evaluation process

## Main comparison protocol

The main unpaired comparison in the paper follows a same-budget continuation setup:

- `CycleGAN baseline`: continuation from `epoch_200` to `epoch_250`
- `MP-CycleGAN`: same initialization and same additional training budget, with training-time multi-perceptual regularization

The method is therefore presented as a controlled refinement strategy rather than a new inference-time backbone.

## Evaluation summary

- `EUVP (in-domain)`: matched subset size `515`
- `UIEB (cross-domain)`: matched subset size `841`
- For paired UIEB PSNR/SSIM analysis, `23` rows with missing full-reference values in both CSV files are excluded, leaving `818` valid paired samples

The paired statistics reported in the paper include:

- mean delta
- median delta
- win rate
- Wilcoxon signed-rank test

## Release note

This public directory focuses on manuscript-side materials and reproducibility helpers. Editorial submission documents and temporary build artifacts are intentionally excluded.
