# Unsupervised and Semi-Supervised Anomaly Detection on the LHC Olympics 2020 R&D Dataset

This repository contains a reproducible, simulation-only benchmark on the public LHC Olympics 2020 R&D samples (Zenodo v5, DOI [10.5281/zenodo.6466204](https://doi.org/10.5281/zenodo.6466204)). It studies one-class and few-label anomaly ranking, training-signal contamination, transfer to a held-out signal topology, and the effect of score selections on the background dijet-mass spectrum. It makes no discovery or real-data sensitivity claim.

## Methods

- **M1:** MLP autoencoder
- **M2:** Isolation Forest
- **M3:** Deep SVDD
- **M4:** Deep SAD with scarce labeled signal events
- **M5:** supervised MLP reference using all labeled training events
- **M6:** Deep SVDD with dijet mass included as a leakage control only

The main detector inputs are `mJ1`, `dmJ`, `tau21_J1`, `tau21_J2`, and `dRJJ`. The resonant observable `mJJ` is excluded from M1-M5 and included only in M6. The full experimental protocol and its limitations are in [PRD_final.md](PRD_final.md); implementation details and measured results are in [reports/technical_note.md](reports/technical_note.md).

## Reproduce

Use Python 3.11 or 3.13 and install the project dependencies (the full reproduction run was executed with Python 3.13; the Dockerfile uses Python 3.11):

```bash
python -m pip install -e ".[dev]"
python scripts/reproduce.py
```

The reproduction command verifies the four official Zenodo files, builds the feature tables, creates deterministic 60/20/20 event splits, runs the contamination and Deep SAD label-budget grids, evaluates held-out 3-prong events, and writes results and diagnostics under `reports/`. The official files total about 3.2 GB. To build the canonical compact feature tables without reclustering every event, it uses the authors' supplied high-level features and validates a deterministic raw-event sample with FastJet. `python scripts/build_features.py --from-raw` instead reclusters every event using the documented exclusive-kT subjettiness axes; that tau definition differs from the released FastJet-contrib values and is not the canonical benchmark input.

For a containerized run, build and run the included Docker image. Raw data are downloaded at runtime and are not included in the image.

## Outputs

- `data/manifest.yaml`: Zenodo checksums, SHA256 digests, file sizes, and schema version
- `data/splits/`: event-level split indices and split manifest
- `reports/feature_validation_*.json`: raw-to-high-level cross-check summary
- `reports/tables/results.csv`: per-run held-out metrics
- `reports/tables/replicate_summary.csv`: mean and standard deviation for the repeated-seed headline comparisons
- `reports/figures/`: background mass, binned acceptance, and score-versus-mass plots
- `reports/tables/experiment_samples/`: exact injected and labeled signal event indices by run
- `output/pdf/LHC_Olympics_Project_Report.pdf`: detailed project report

The fixed-window bump diagnostic uses the 3.3-3.7 TeV window and sidebands; it is not a global search significance. AUC, rejection, and SIC are simulated-sample ranking metrics, not physics sensitivity.

