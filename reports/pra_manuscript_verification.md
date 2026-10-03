# APS/PRA-format manuscript verification

Verified on 3 October 2026 against numerical release v1.0.1, commit `4b15306cfac9e3f57ace098932db0e4ea69e0f2a`. This manuscript is a subsequent documentation artifact and does not change the frozen numerical release.

## Delivered files

- `lhco2020_pra.tex`: REVTeX 4.2, `reprint,aps,pra`, self-contained numerical tables and bibliography; existing released figures are required for compilation.
- `final/lhco2020_pra.pdf`: 9 pages, 6 tables, 2 multipanel figures, 16 cited references; 538,959 bytes.
- `pra_data_audit.json`: six five-seed headline summaries, model parameter counts, seed-42 bootstrap intervals, two paired comparisons, and the 23-configuration single-seed grid.
- `novelty_and_journal_assessment.md`: ten-paper primary-literature shortlist, scope/readiness verdict, limitations, and proposed scientific follow-up.

PDF SHA-256: `8bdc0fb97ebb5354dda4ef01576db41bb8b863615deba5115ead70cad68d26f1`.

## Numerical and scientific checks

Headline summaries were calculated from `tables/results.csv`, requiring exactly seeds 42–46 at each headline setting. Standard deviations use the sample convention. Conditional intervals and paired differences were read from `tables/auc_bootstrap.json`; signs and reversed interval endpoints were checked for the SAD-minus-Isolation-Forest comparison. All four contamination rows, nine label-grid cells, and ten local-fit rows were populated directly from the canonical tables. The failed SAD seed-44 1% fit is retained.

Parameter counts were calculated from the actual core architectures. The training implementation was inspected for scaler fitting, labelled-event use, center initialization, minibatch losses, early stopping, and optimization settings. The supervised `k1000` metadata field is explicitly distinguished from its actual use of all training signal labels. The density proxies are distinguished from the published ANODE/CATHODE algorithms. Particle and calibration measurements retain their archived status and original sample sizes. Figure generation was checked: ROC panels are seed 42; transfer/JSD panels show the five headline seeds and means.

Primary literature and APS scope/policy records were checked on the date above. Two September/June 2026 items in the novelty shortlist remain labelled preprints; the optimal-transport paper's published PRD volume/article and ANTELOPE author names were checked against primary records. Author/affiliation details are carried forward from the existing manuscript. Funding/contribution confirmation and independent human scientific review remain pending.

## Build and presentation checks

The built-in source editor was opened. Its compiler returned `Unable to find standard directories for platform`, so the installed MiKTeX compiler produced the PDF. Final compilation completed with no undefined citations/references, overfull boxes, or stuck floats. Ordinary underfull-line and package compatibility warnings do not affect the inspected output. All nine pages were rendered and visually inspected; the final changed caption page was checked again. All 16 citation keys resolve to bibliography entries, and numerical template markers are absent.

To rebuild from the repository root with an installed REVTeX-capable TeX distribution:

```text
pdflatex --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=output/pdf reports/lhco2020_pra.tex
pdflatex --disable-installer -interaction=nonstopmode -halt-on-error -output-directory=output/pdf reports/lhco2020_pra.tex
```

The existing release's 30 passing tests and full core reproduction are recorded in `release_status.md` and `reproduction/`; no model code or result artifacts changed during this manuscript task, so those experiments/tests were not repeated. LaTeX compilation and numerical/presentation checks are the relevant verification for these documentation changes. Software release completion, manuscript-format compliance, scientific novelty, and journal acceptance remain separate claims.
