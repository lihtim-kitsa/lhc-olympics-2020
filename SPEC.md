# Release completion

## Goal
Close the release-tooling gaps without changing the frozen physics protocol or existing results.

## Scope and files
- Add meaningful feature, split, scaler-isolation, label-disjointness and threshold tests under `tests/`.
- Add `scripts/reproduce_fast.py` and `make reproduce-fast`: a seeded, cached-feature Isolation Forest demo in an isolated output directory.
- Fix packaging only if fresh-checkout installation exposes a failure.
- Document verification and outstanding acceptance requirements in `reports/release_status.md` and README.

## Acceptance
- Tests pass and changed Python files pass lint and syntax checks.
- Fast reproduction trains and evaluates from cached features without changing canonical artifacts, with recorded sample sizes, seed, versions and runtime.
- A fresh Git checkout can install the project and run the tests and demo using the local dependency environment; distinguish this from a new-environment full-grid reproduction.
- Check five-seed coverage and public-repository status before tagging v1.0. Never label an incomplete benchmark as a final release.
- Preserve existing manuscript edits and generated results; commit release tooling separately.
