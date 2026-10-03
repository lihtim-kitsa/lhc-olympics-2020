# v1.0.1 release acceptance — 3 October 2026

The benchmark release is complete within the declared simulation-only scope. This is author verification, not an external scientific review or a discovery claim.

## Verified

- All 47 frozen core configurations trained and evaluated from scratch in a fresh checkout: the complete seed-42 grid plus six headline configurations repeated over seeds 43–46.
- Isolated Python 3.11.9 dependency installation; raw HDF5 Blosc compression explicitly registered through hdf5plugin. Pinned Windows verification environment: `requirements-release-py311.txt`.
- Four official raw files checksum-verified and shared read-only; features and splits rebuilt. No previous core model checkpoint was reused. The raw downloads themselves were not repeated.
- Full reproduction entrypoint completed after staged training, retaining all 47 verified fits and regenerating summaries. Three serial fits and 44 fits across three independent worker stores are recorded separately.
- 30 tests pass, covering analytic feature cases, extended schema, split disjointness, scaler isolation, label isolation, metrics, tied thresholds, experiment scheduling/resume and paired AUC uncertainty. Release-tooling lint, wheel build and diff checks pass.
- 500 stratified paired-bootstrap trials on frozen seed-42 test scores; 199,984 background and 19,996 signal events. AUC intervals and all pairwise differences are separate from training-seed variance. Saved scores include matching event IDs.
- Manuscript tables and figures generated from the final table; PDF compilation resolves citations. Reference identity audit and visual PDF checks completed.
- Public repository verified: https://github.com/lihtim-kitsa/lhc-olympics-2020 . Final source, results and PDFs committed for the annotated `v1.0` tag.

## Results

Five-seed mean ± sample standard deviation; the three-prong signal is held out during training.

| Model | 2-prong AUC | Held-out 3-prong AUC |
|---|---:|---:|
| M1_Autoencoder | 0.7175 ± 0.0741 | 0.5965 ± 0.1026 |
| M2_IsolationForest | 0.8068 ± 0.0077 | 0.6831 ± 0.0206 |
| M3_DeepSVDD | 0.5002 ± 0.0072 | 0.6170 ± 0.0114 |
| M4_DeepSAD | 0.8880 ± 0.0114 | 0.7492 ± 0.0614 |
| M5_Supervised | 0.9659 ± 0.0002 | 0.9292 ± 0.0012 |
| M6_MassAware | 0.5151 ± 0.0105 | 0.6670 ± 0.0233 |

The final core table uses the clean Python 3.11 CPU run consistently. Historical measurements are preserved in `reports/reproduction/historical_results.csv`; they differ for some neural configurations. `reports/reproduction/verification.json` quantifies the differences. Five exploratory baselines retain their separately evaluated single-seed results and are not represented as repeated core experiments.

## Runtime and disk footprint

The staged data-verification, feature/split rebuilding, full training/evaluation and resume check took 33.3 minutes of wall time, including staging pauses; raw downloading was excluded. The three-worker stage trained/evaluated 44 configurations in 24.8 minutes, following three initial serial fits. The final resume entrypoint took 15.8 seconds. Raw inputs occupy 3.223 GB; regenerated feature caches 61.6 MB; core model artifacts 10.5 MB; diagnostic figures 11.5 MB. Temporary environments and duplicate worker stores require additional space. Peak RAM was not measured.

## Declared protocol deviations and limits

- Canonical inputs use released high-level tau values. Raw clustering is cross-checked on 128 events; its exclusive-kT tau convention differs from the supplied FastJet-contrib convention. This is not exact full raw feature reproduction.
- Every stochastic core method has five headline seeds; contamination/label sweeps and exploratory extensions primarily use seed 42.
- Finite-test intervals cover seed-42 AUC only, not every reported metric. The 50-trial null ensemble is limited; separate 500-trial closure studies do not establish global significance or detector-systematic closure.
- Three-prong generalisation uses frozen two-prong-trained scores. A matched three-prong training study, full systematic model and external independent review remain future research.
- No retrospective claim is made that an evaluation-free release-candidate tag existed before the earlier exploratory tests. The historical evaluation sequence is preserved.
- The pinned environment is Windows/Python 3.11 specific. Docker build and Linux reproduction were not independently run. No claim of exact neural-score identity across platforms, dependency versions or CPU reduction schedules.

## Verify

```text
python -m pip install -e ".[dev]"
python -m pytest tests/ -q -p no:cacheprovider
python scripts/reproduce_fast.py
python scripts/reproduce.py
python scripts/bootstrap_auc.py
python scripts/update_paper_results.py
python scripts/build_report.py
```

For the pinned reference stack, install `requirements-release-py311.txt` in a Python 3.11 Windows environment before installing the editable project. Full reproduction requires approximately 3.2 GB of raw input plus output/environment storage. Release PDFs are under `reports/final/`.

The v1.0.1 packaging patch records checksums for exact Git-archive bytes; v1.0 was preserved. Numerical results and PDFs are unchanged.
