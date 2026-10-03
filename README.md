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

For quick verification using cached features and frozen splits, run `make reproduce-fast`
or `python scripts/reproduce_fast.py`. This seeded Isolation Forest demo samples up to
2,000 events per split, fits scaling on training background, and calibrates its cut
on validation background. It writes metrics, versions, runtime and scores under
`output/reproduce_fast/`; its results are separate from the full benchmark.
Run `python -m pytest tests/ -q -p no:cacheprovider` for protocol checks.
Release acceptance and outstanding requirements are recorded in
[reports/release_status.md](reports/release_status.md).

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

## Extended studies

- **Particle-level representation:** `python scripts/train_particle_deepsets.py data/raw/events_anomalydetection_v2.h5 --hypotheses lhco_3prong --max-train-events 50000 --epochs 8 --output-dir output/particle_deepsets_final`. This trains a masked, permutation-invariant Deep Sets autoencoder on 2-prong background only, chooses a threshold on validation background, and scores held-out hypotheses directly from raw HDF5 in batches. It avoids writing a padded 15 GB intermediate. Its reconstruction score is a ranking score, not a likelihood. To evaluate additional raw samples against a saved checkpoint without retraining, run `python scripts/evaluate_particle_hypotheses.py data/raw/events_anomalydetection_v2.h5 output/particle_deepsets_final/particle_deepsets.pt --hypotheses xtoyyprime,xtowrto3w` (the default comparison uses a seeded 20k held-out-background sample).
- **Additional signal hypotheses:** available IDs and provenance are listed in `configs/signal_hypotheses.yaml`. Four supplementary hypotheses are in [Zenodo record 18983506](https://zenodo.org/records/18983506); fetch chosen archives with `python scripts/fetch_signal_hypotheses.py xtowrto3w xtoyyprime`. The two reported archives passed their published checksums; their large event files are excluded from Git. Supplementary samples use a newer generation chain, so report them as external evaluations, separate from the original R&D benchmark.
- **Density-score calibration:** after training M8 or M11, run `python scripts/evaluate_density_calibration.py --config configs/m8_anode.yaml` or `--config configs/m11_cathode_cvae.yaml`. ANODE now splits held-out sideband background into separate conformal calibration/evaluation halves and reports calibrated upper-tail NLL p-values plus raw GMM PIT diagnostics. These p-values apply to exchangeable sideband background; they do not establish an absolute or signal-region conditional density. CATHODE reports calibration of the held-out real-vs-generated classifier, not an absolute density.
- **Background-only closure:** `python scripts/background_closure.py --config configs/m4_deep_sad.yaml --trials 500` freezes the detector and validation-selected thresholds, then repeats the sideband fit on Poisson-bootstrap pseudoexperiments from held-out background. Score ties at the operating point use fractional boundary acceptance, so validation efficiencies remain 10%/1% even for saturated scores. Outputs go to `reports/closure/`. This measures local fit/selection behavior under resampling; it does not cover detector systematics, alternate background models, or global significance.

These extension scripts write separate artifacts and do not alter the original benchmark outputs. They require the corresponding local checkpoints and feature/data files; external signal archives are fetched only when explicitly requested by ID.

