# Release acceptance status

Checked 2026-10-03. Release-tooling checkpoint: `0b441f9`.

## Verified

- 25 tests pass in the working checkout and a fresh local Git clone. Coverage includes analytic jet-feature cases, deterministic/disjoint splits, training-only scaling, disjoint Deep SAD signal labels, metrics and validation-calibrated cuts including score ties.
- A loader bug was fixed: `load_data` now uses its supplied canonical feature-file path. Default benchmark paths are unchanged.
- `make reproduce-fast` / `python scripts/reproduce_fast.py` runs a seeded cached-feature Isolation Forest demo independently of canonical results.
- Demo seed 42: 2,000 events per split, AUC 0.81020475, validation background acceptance 0.1000, test background acceptance 0.0945 and test signal acceptance 0.3820. Fresh-clone metrics match; model fitting and scoring took approximately 0.5 seconds, excluding interpreter/import startup.
- Offline wheel build and isolated package installation pass in both checkouts. Package discovery explicitly includes `src` and its subpackages.
- Changed/new standalone Python files pass Flake8 (E501/W503 excluded to match existing wrapping); changed Python files parse successfully.
- Saved background/2-prong splits contain 1,099,893 events without overlap. Cached feature HDF5 files total 58,465,336 bytes; raw files total 3,223,240,722 bytes.

## Verification environment and boundaries

Windows, Python 3.13.9; NumPy 2.3.5, h5py 3.15.1 and scikit-learn 1.7.2. Tests use the existing installed dependency environment. Fresh-clone package installation used `--no-deps --no-build-isolation`, so it does not prove dependency resolution on a new machine. The clone was local, with committed cached features; no new raw download or full-grid retraining was performed. Demo outputs are ignored under `output/reproduce_fast/`.

For this Windows sandbox, temporary pytest files were placed inside the workspace using a new `--basetemp` directory; the default sandbox temporary location was inaccessible. Do not reuse a basetemp path containing files you want to retain, because pytest clears it.

## Outstanding final-release requirements

1. Five-seed experiments: the saved table has 40 unique rows, including 35 M1–M6 rows. Each M1–M6 method currently has seeds 42–44; the full contamination/label-budget grid is only seed 42. The existing full reproduction command schedules seeds 42–46 and retains completed checkpoint/result pairs. Its aggregate currently selects only seeds 42–44, which must be corrected and the report updated when new runs finish.
2. Full reproduction from a fresh environment, including dependency installation, raw-file verification, every required experiment and measured end-to-end runtime/disk footprint. The successful fast demo is narrower evidence.
3. Review and validate the existing uncommitted LaTeX manuscript and bibliography changes, and reconcile report claims with the final result table. Preserve extended-study documentation when regenerating the report: the current report builder rewrites `technical_note.md` from the core table.
4. Confirm public GitHub accessibility. The configured remote is `https://github.com/lihtim-kitsa/lhc-olympics-2020`; the web check did not establish visibility.
5. Commit validated final report artifacts and create `v1.0` only after the above checks succeed. No release tag was created at this checkpoint.

## Commands

```text
python -m pytest tests/ -q -p no:cacheprovider
python scripts/reproduce_fast.py
python scripts/reproduce.py
```

The first two are quick verification. The last runs the full experiment matrix and requires the complete dependency/data environment.
