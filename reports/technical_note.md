# Technical Note: Unsupervised and Semi-Supervised Anomaly Detection on the LHC Olympics 2020 R&D Dataset

## Status and scope

The simulation-only benchmark contains 35 unique result configurations over seeds 42, 43, 44. The full contamination/label-budget matrix is seed 42; repeated seeds 43 and 44 cover six headline configurations. No discovery or real-data sensitivity claim is made.

## Methods and data

M1 is an MLP autoencoder; M2 Isolation Forest; M3 Deep SVDD; M4 Deep SAD; M5 a supervised MLP reference; M6 a mass-aware Deep SVDD leakage control. Inputs are mJ1, dmJ, tau21 for each leading jet, and dRJJ. mJJ is excluded from M1–M5 and included only in M6. Official Zenodo v5 high-level features are canonical; raw anti-kT/exclusive-kT validation is a deterministic 128-event check. The local tau axes do not exactly reproduce the supplied FastJet-contrib convention.

## Measured seed-42 headline results

| Model | f_train | k | 2-prong AUC | 3-prong AUC | JSD at 10% |
|---|---:|---:|---:|---:|---:|
| M1_Autoencoder | 0% | 0 | 0.6541 | 0.5894 | 0.1375 |
| M2_IsolationForest | 0% | 0 | 0.8165 | 0.7144 | 0.0232 |
| M3_DeepSVDD | 0% | 0 | 0.4830 | 0.6224 | 0.0011 |
| M4_DeepSAD | 50% | 100 | 0.9335 | 0.8324 | 0.0149 |
| M5_Supervised | 0% | 1000 | 0.9659 | 0.9275 | 0.0070 |
| M6_MassAware | 0% | 0 | 0.5395 | 0.6651 | 0.0259 |

Metrics are ranking/shape diagnostics on simulated test data. The held-out 3-prong sample is never used to train these configurations. Thresholds are selected on validation background and frozen for test evaluation. The fixed-window sideband diagnostic is local; its 50-trial Poisson bootstrap has coarse tail resolution and is not a global p-value.

## Reproducibility and limitations

Run `python -m pip install -e ".[dev]"` followed by `python scripts/reproduce.py`. The run verifies the official data files, builds features and deterministic splits, evaluates the grid, and writes model/data/run metadata under `data/`, `models/`, and `reports/`. Regenerate the PDF with `python scripts/build_report.py`.

Remaining limitations include simulation-only scope, the released/local tau definition mismatch, only three seeds for headline runs, a seed-42 full grid, the limited null bootstrap ensemble, simplified fixed-window sideband fitting, and lack of a clean-room independent reproduction. See the PDF report and `PRD_final.md` for full protocol and audit detail.
