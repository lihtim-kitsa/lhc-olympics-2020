# Anomaly Detection on the LHC Olympics 2020 R&D Dataset

This repository implements a rigorous, reproducible benchmark for unsupervised and semi-supervised anomaly detection on the LHC Olympics 2020 R&D dataset (Zenodo v5).

## Methods
- Deep SVDD (M3)
- Deep SAD (M4)
- Isolation Forest (M2)
- Autoencoder (M1)
- Supervised Reference (M5)

## Reproduction
To reproduce the pipeline:
```bash
make env
make data
make features
make train
make evaluate
```

## Structure
- `PRD_final.md`: Pre-registered experiment protocol.
- `src/`: Data processing, models, and evaluation logic.
- `configs/`: Hyperparameters and run settings.
- `scripts/`: Execution scripts.

*Simulation-only benchmark. No physical discovery claimed.*
