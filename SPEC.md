# Release completion

## Expand the original draft in the open manuscript (2026-10-03)
- Goal: build on the user's supplied original draft rather than replace its research story; revise `reports/lhco2020_pra.tex` in place and retain APS/PRA formatting.
- Scope: restore the original title, four questions, model/feature overview, practical interpretation, failure diagnostics, and MD-SWAD/extended-feature ideas with accurate evidence status; retain released numbers and correct unsupported historical claims.
- Files: existing open LaTeX source and documentation of this revision; do not create a replacement source, a separately compiled PDF, or another editor tab.
- Acceptance: every new result traces to existing artifacts; theoretical ingredients cite primary sources; explicitly distinguish candidate methods from validated results; call the built-in compiler and report infrastructure limitations honestly.

## Academic manuscript and novelty assessment (2026-10-03)
- Goal: write a complete research manuscript about the released benchmark in Physical Review A REVTeX format and assess novelty, validity and journal fit against primary literature.
- Files: new `reports/lhco2020_pra.tex`, `reports/novelty_and_journal_assessment.md`, `reports/pra_data_audit.json`, verification notes, and a visually verified manuscript PDF under `reports/final/`; preserve existing reports, numerical results and release tags.
- Scope: final core tables, statistical definitions, exploratory extensions, reproducibility and declared deviations; primary-source comparison limited to ten key works. No new experiments or invented quantum component.
- Acceptance: numerical claims trace to released artifacts; distinguish seed variance from finite-test intervals; verify official APS scope/style; inspect methodological implementations; compile with resolved citations and inspect every page.
- Delivery: editable LaTeX, compiled PDF and a candid submission-readiness verdict. Do not submit or alter public tags. Use the built-in editor/compiler where supported and preserve source on native compiler failure.

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

## Final closeout (2026-10-03)
- Complete five-seed M1–M6 headline comparisons while retaining the seed-42 contamination/label sweep, without changing training or evaluation choices.
- Correct summary seed coverage, resume metadata and percentage labels; retain extended technical-note content.
- Validate manuscript tables/figures against saved results, compile and visually inspect the PDF.
- Verify in an isolated dependency environment; record environment, wall time, disk use and limitations.
- Report stratified paired-bootstrap AUC intervals on frozen seed-42 test events separately from training-seed variance; validate weighted tie handling against analytic and sklearn cases.
- Commit the final source/results, confirm the public remote, and tag/publish v1.0 only when acceptance passes.
