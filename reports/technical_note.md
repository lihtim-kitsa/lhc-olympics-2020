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

## Extended-study implementation and preliminary closure

- A six-channel constituent representation (`pt/sum(pt)`, `log(pt/max pt)`, `log1p(pt)`, scaled eta, sin(phi), cos(phi)) and masked permutation-invariant Deep Sets autoencoder are implemented. The expanded run trained 8 epochs on a seeded 50,000-event subset of the 599,952-event background training split, selected its threshold on all 199,984 validation events, and scored all 199,984 test-background, 19,996 2-prong, and 100,000 held-out 3-prong events. AUCs were 0.498 (2-prong) and 0.441 (3-prong); signal efficiencies at the validation 1% threshold were 0.160% and 0.098%, respectively. The model therefore shows no useful separation in this run. The shared split indices were used, and the best validation-loss epoch was retained.
- The additional-signal registry includes the original 2-prong and held-out 3-prong R&D samples plus four supplementary models from Zenodo 18983506. Two external archives have been checksum-verified and scored with the frozen Deep Sets checkpoint. Against the same seeded 20,000-event held-out background subset, `XtoYYprime` (44,673 events) has AUC 0.465 and 0.159% efficiency at the validation 1% threshold; `XtoWRto3W` (32,801 events, 2+4 prongs) has AUC 0.507 and 0.280% efficiency. These near-random external results use a different simulation chain and are reported separately from the original R&D benchmark.
- ANODE's held-out sideband-GMM score was split equally into calibration and evaluation halves (64,056 events each). Split-conformal upper-tail NLL p-values had KS statistic 0.0043 (p=0.179); observed rates below 1%, 5%, and 10% were 1.054%, 4.975%, and 9.868%. This calibrates anomaly-score p-values for exchangeable sideband background; it does not produce an absolute likelihood or validate transfer into the signal region. Raw GMM Rosenblatt PIT remains a separate density-fit diagnostic. CATHODE's held-out real-vs-generated classifier has Brier score 0.0087 and ECE 0.0034 for the balanced evaluation mixture, which calibrates only that classifier task—not absolute background density.
- Tie-aware operating points now attain exactly 10%/1% expected validation acceptance, including models with saturated or tied scores. With 500 Poisson-bootstrap trials and sideband-fit quality rejection, post-1% local Z>=3 rates were 0/461 for M1, 0/479 for M2, and 1/456 for M11. This is a conditional local closure diagnostic on the released simulation, not full detector/systematic closure or global significance.

## Reproducibility and limitations

Run `python -m pip install -e ".[dev]"` followed by `python scripts/reproduce.py`. The run verifies the official data files, builds features and deterministic splits, evaluates the grid, and writes model/data/run metadata under `data/`, `models/`, and `reports/`. Regenerate the PDF with `python scripts/build_report.py`.

Remaining limitations include simulation-only scope, the released/local tau definition mismatch, only three seeds for headline runs, a seed-42 full grid, the limited null bootstrap ensemble, simplified fixed-window sideband fitting, and lack of a clean-room independent reproduction. See the PDF report and `PRD_final.md` for full protocol and audit detail.
