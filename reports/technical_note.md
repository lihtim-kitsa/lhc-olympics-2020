# Technical Note: Unsupervised and Semi-Supervised Anomaly Detection on the LHC Olympics 2020 R&D Dataset

## Status and scope

The simulation-only benchmark contains 52 unique result configurations over seeds 42--46. The contamination/label-budget matrix is seed 42; five seeds cover the six core headline configurations. Supplemental baselines remain exploratory single-seed evaluations. The frozen reproduction schedule has 47 core runs; the table also contains five supplemental baselines. No discovery or real-data sensitivity claim is made.

## Methods and data

M1 is an MLP autoencoder; M2 Isolation Forest; M3 Deep SVDD; M4 Deep SAD; M5 a supervised MLP reference; M6 a mass-aware Deep SVDD leakage control. Inputs are mJ1, dmJ, tau21 for each leading jet, and dRJJ. mJJ is excluded from M1–M5 and included only in M6. Official Zenodo v5 high-level features are canonical; raw anti-kT/exclusive-kT validation is a deterministic 128-event check. The local tau axes do not exactly reproduce the supplied FastJet-contrib convention.

## Final core results

| Model | 2-prong AUC | Held-out 3-prong AUC |
|---|---:|---:|
| M1_Autoencoder | 0.7175 ± 0.0741 | 0.5965 ± 0.1026 |
| M2_IsolationForest | 0.8068 ± 0.0077 | 0.6831 ± 0.0206 |
| M3_DeepSVDD | 0.5002 ± 0.0072 | 0.6170 ± 0.0114 |
| M4_DeepSAD | 0.8880 ± 0.0114 | 0.7492 ± 0.0614 |
| M5_Supervised | 0.9659 ± 0.0002 | 0.9292 ± 0.0012 |
| M6_MassAware | 0.5151 ± 0.0105 | 0.6670 ± 0.0233 |

Five-seed mean ± standard deviation. All 47 core results come from the fresh Python 3.11 CPU run. Historical results remain in `reports/reproduction/historical_results.csv`. Seed-42 sweeps and single-seed exploratory extensions are distinguished from these averages.

## Extended-study implementation and preliminary closure

- A six-channel constituent representation (`pt/sum(pt)`, `log(pt/max pt)`, `log1p(pt)`, scaled eta, sin(phi), cos(phi)) and masked permutation-invariant Deep Sets autoencoder are implemented. The expanded run trained 8 epochs on a seeded 50,000-event subset of the 599,952-event background training split, selected its threshold on all 199,984 validation events, and scored all 199,984 test-background, 19,996 2-prong, and 100,000 held-out 3-prong events. AUCs were 0.498 (2-prong) and 0.441 (3-prong); signal efficiencies at the validation 1% threshold were 0.160% and 0.098%, respectively. The model therefore shows no useful separation in this run. The shared split indices were used, and the best validation-loss epoch was retained.
- The additional-signal registry includes the original 2-prong and held-out 3-prong R&D samples plus four supplementary models from Zenodo 18983506. Two external archives have been checksum-verified and scored with the frozen Deep Sets checkpoint. Against the same seeded 20,000-event held-out background subset, `XtoYYprime` (44,673 events) has AUC 0.465 and 0.159% efficiency at the validation 1% threshold; `XtoWRto3W` (32,801 events, 2+4 prongs) has AUC 0.507 and 0.280% efficiency. These near-random external results use a different simulation chain and are reported separately from the original R&D benchmark.
- ANODE's held-out sideband-GMM score was split equally into calibration and evaluation halves (64,056 events each). Split-conformal upper-tail NLL p-values had KS statistic 0.0043 (p=0.179); observed rates below 1%, 5%, and 10% were 1.054%, 4.975%, and 9.868%. This calibrates anomaly-score p-values for exchangeable sideband background; it does not produce an absolute likelihood or validate transfer into the signal region. Raw GMM Rosenblatt PIT remains a separate density-fit diagnostic. CATHODE's held-out real-vs-generated classifier has Brier score 0.0087 and ECE 0.0034 for the balanced evaluation mixture, which calibrates only that classifier task—not absolute background density.
- Tie-aware operating points now attain exactly 10%/1% expected validation acceptance, including models with saturated or tied scores. With 500 Poisson-bootstrap trials and sideband-fit quality rejection, post-1% local Z>=3 rates were 0/461 for M1, 0/479 for M2, and 1/456 for M11. This is a conditional local closure diagnostic on the released simulation, not full detector/systematic closure or global significance.

## Reproducibility and limitations

Run `python -m pip install -e ".[dev]"` followed by `python scripts/reproduce.py`. The run verifies the official data files, builds features and deterministic splits, evaluates the grid, and writes model/data/run metadata under `data/`, `models/`, and `reports/`. Regenerate the PDF with `python scripts/build_report.py`.

Remaining limitations include simulation-only scope, the released/local tau definition mismatch, five-seed uncertainty only for core headline settings, predominantly single-seed contamination/label-budget sweeps, the limited null bootstrap ensemble and simplified fixed-window sideband fitting. Finite-test AUC intervals are recorded separately in `reports/tables/auc_bootstrap.json`: 500 stratified paired bootstrap trials (seed 20261003) on 199,984 background and 19,996 signal events, from the final saved seed-42 checkpoints. These intervals are conditional on the fitted detectors and are separate from seed variance; other-metric intervals, detector systematics and external independent review remain outside this release. Clean-checkout verification is recorded in `reports/release_status.md`.

The extended feature file was rebuilt with schema 2.0.0: the first five columns match the canonical core inputs, followed by tau32 for each jet, mJJ and the label. The previous cached file placed dRJJ after the tau32 columns. M9/M10 numerical claims are excluded pending a matching evaluation table; this does not affect M1--M6, which use the canonical feature file.
