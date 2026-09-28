# PRD: Unsupervised and Semi-Supervised Anomaly Detection on the LHC Olympics 2020 R&D Dataset

| | |
|---|---|
| **Author** | Mithil Hardik Astik |
| **Version** | 2.0 |
| **Date** | 28 September 2026 |
| **Working repo name** | `lhco-anomaly-svdd` |
| **Time budget** | 4 weeks (28 Sep – 25 Oct 2026), part-time alongside APEX-PINN |
| **Status** | Ready to start |
| **Revision** | Methodological tightening of the v1.0 draft: dataset provenance, feature definition, leakage controls, statistical protocol, null test, operating points, and reproducibility |

---

## 1. Summary

Build a small, fully reproducible benchmark that applies Deep SVDD (unsupervised) and Deep SAD (semi-supervised) anomaly detectors to the public LHC Olympics 2020 R&D dijet dataset, and compare them against simple baselines and a supervised reference under the same leakage-safe protocol.

The project reuses the detector ideas from my TF-QKD zero-day work (one-class detection, benign contamination versus anomalous events, held-out anomaly types) in a collider setting. It is a **simulation-only methods benchmark**. It makes no discovery claim, and detector metrics are not presented as physics sensitivity.

The central research question is:

> **How do one-class and few-label anomaly detectors trade off detection performance, robustness to contaminated training data, generalization to an unseen signal topology, and preservation of the resonant background spectrum?**

The benchmark is intentionally narrow. It prioritizes a rigorous, reproducible experiment over a large number of increasingly sophisticated architectures.

## 2. Background and motivation

- Searches for new physics at the LHC often target specific models. Anomaly detection instead aims to identify events that differ from the background without requiring the search to be optimized for a single signal model.
- The LHC Olympics 2020 R&D dataset provides simulated QCD dijet background and a known 2-prong signal for development, plus a separate 3-prong signal release for generalization studies. The current Zenodo v5 R&D record is `10.5281/zenodo.6466204`. It contains 1M QCD background events and 100k 2-prong signal events, with an additional 100k 3-prong signal sample. The events were generated with Pythia8 and Delphes 3.4.1, with no pileup or MPI, and selected with a single fat-jet trigger (`R=1`, pT threshold 1.2 TeV). The raw representation contains up to 700 massless reconstructed particles per event plus the development truth bit, giving 2101 stored values per event.
- A central practical problem is **mass sculpting**: an anomaly score correlated with the dijet mass `mjj` can distort the background mass spectrum so that a detector cut creates a fake resonant excess.
- The dataset itself provides a truth bit for testing and development. The benchmark may use it to construct controlled training regimes, but the truth bit must never be used to tune unsupervised methods in a way that leaks test information.

**Why this project, for me:** it is finishable in a month, runs on a laptop or free Colab (no HPC dependency), reuses my Deep SVDD/SAD pipeline, and gives a public finished result alongside my in-progress work. It also exercises tooling relevant to CERN: HEP data formats, jet reconstruction, statistical treatment of a resonant bump, and reproducible ML experimentation.

## 3. Goals

| ID | Goal |
|---|---|
| G1 | Reproduce a credible anomaly-detection benchmark on the LHCO R&D dataset from raw download to results table with one command. |
| G2 | Compare Deep SVDD and Deep SAD against an autoencoder, Isolation Forest, and a supervised reference under the same leakage-safe protocol and feature budget. |
| G3 | Quantify robustness to signal contamination in the training data and quantify how a small labelled-anomaly budget changes Deep SAD performance. |
| G4 | Evaluate generalization from the 2-prong training signal to a held-out 3-prong signal topology, if the raw-data preprocessing path fits the schedule. |
| G5 | Measure mass sculpting, score–`mjj` dependence, and fixed-window bump-hunt behaviour before and after anomaly-score cuts. |
| G6 | Run a background-only null experiment to measure the detector's tendency to create artificial mass excesses. |
| G7 | Publish a clean public repo and a short technical write-up whose claims match the evidence exactly. |

## 4. Non-goals

- No claim of discovering, or being sensitive to, real new physics.
- No claim of state-of-the-art performance.
- No new detector architecture in the core project.
- No full detector simulation and no ATLAS/CMS proprietary data.
- No HPC dependency.
- No use of the challenge black-box datasets in the core four-week scope.
- No broad survey of modern HEP anomaly-detection methods. More sophisticated density-estimation, flow, transformer, graph, or decorrelation methods remain outside the core scope unless added later as stretch work.

## 5. Users and stakeholders

- **Primary:** me, as evidence for the CERN STAG 2026 application and later applications.
- **Secondary:** a CERN supervisor or recruiter skimming the repo in a few minutes, so the README must state the problem, method, dataset, and headline result within the first screen.
- **Tertiary:** my advisor (T S L Radhika) as an optional reviewer.

## 6. Data

### 6.1 Source and exact dataset version

- Primary source: **LHC Olympics 2020 R&D dataset, Zenodo v5, DOI `10.5281/zenodo.6466204`**.
- Original challenge paper: Kasieczka et al., *The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics*, arXiv:2101.08320.
- The selected Zenodo record contains the updated v2 raw background file, the 2-prong signal, the 3-prong signal, high-level feature files, the Delphes card, and the Pythia command files.
- On day 1, confirm the exact downloaded filenames and record the MD5/SHA256 checksum of every downloaded file in the repository.
- Record the Zenodo DOI, record version, download date, filename, byte size, checksum, and local feature-schema version in a machine-readable manifest such as `data/manifest.yaml`.
- Raw data is never committed to git.

### 6.2 Files used in the core and generalization experiments

Core 2-prong benchmark:

- `events_anomalydetection_v2.h5`
- `events_anomalydetection_v2.features.h5`

Held-out 3-prong generalization experiment:

- `events_anomalydetection_Z_XY_qqq.h5`
- `events_anomalydetection_Z_XY_qqq.features.h5`

The high-level files are optional inputs for validation and cross-checking, not a substitute for verifying the raw-to-feature pipeline. The main feature-generation path should be reproducible from the raw event representation.

### 6.3 Size and memory

The raw representation is 1.1M events × 2101 values. A naive dense float32 representation is roughly 9 GB in memory, so the pipeline **must read the raw HDF5 data in chunks** and cache only compact derived features.

The reference v5 Zenodo record reports approximately 2.9 GB for the updated 2-prong raw HDF5 file and approximately 235 MB for the 3-prong raw HDF5 file.

### 6.4 Dataset splits

Use deterministic, event-level splits with a fixed master seed.

For the 2-prong benchmark, create separate background and signal pools before constructing the final training/validation/test mixtures:

- Background: 60% train / 20% validation / 20% test.
- 2-prong signal: 60% train / 20% validation / 20% test.

For all controlled contamination experiments:

- Background train events are the base pool.
- Signal contamination is sampled only from the **signal-train pool**.
- The test split is never used for training, hyperparameter selection, threshold selection, or feature-statistic fitting.

For the 3-prong generalization experiment, keep the entire 3-prong evaluation pool disjoint from any 2-prong model training or tuning.

All split indices are generated once, saved, reviewed, and committed before final evaluation.

All preprocessing statistics (scalers, quantiles, clipping parameters if any) are fit on the relevant training split only.

## 7. Technical approach

### 7.1 Research design and experiment terminology

Three quantities must remain distinct throughout the project:

- **Training contamination fraction, `f_train`:** fraction of anomalous signal events injected into the otherwise background-only training pool.
- **Label budget, `k`:** number of explicitly labelled signal events supplied to Deep SAD.
- **Test signal prevalence, `f_test`:** fraction of signal events used when constructing a signal-plus-background evaluation mixture for any prevalence-dependent diagnostic.

Changing one must not silently change another.

### 7.2 Preprocessing

1. Read raw events in chunks.
2. Cluster each event into anti-kT jets with `R = 1` using a FastJet-compatible implementation.
3. Keep the two highest-pT jets and define them explicitly as `J1` and `J2`.
4. Compute the following detector inputs:

   | Feature | Definition | Used by main detectors? |
   |---|---|---|
   | `mJ1` | mass of pT-leading jet | Yes |
   | `dmJ` | `|mJ1 - mJ2|` | Yes |
   | `tau21_J1` | `tau2_J1 / tau1_J1` | Yes |
   | `tau21_J2` | `tau2_J2 / tau1_J2` | Yes |
   | `dRJJ` | angular distance between `J1` and `J2` | Yes |
   | `mJJ` | invariant mass of the dijet system | **No: resonant variable only** |

5. The main detector input vector is therefore exactly:

   ```text
   X = [mJ1, dmJ, tau21_J1, tau21_J2, dRJJ]
   ```

6. `mJJ` must be excluded from M1–M5 main detector inputs to reduce the most direct route to mass sculpting.
7. Jet ordering is by descending jet pT, matching the dataset's documented `j1`/`j2` convention.
8. If `tau1` is numerically zero, the implementation must use a declared handling rule rather than silently producing infinities or NaNs. The preferred rule is to flag the event, quantify the affected fraction on the training split, and exclude or repair it using a fixed documented convention before model fitting.
9. Cache the resulting feature table to Parquet or HDF5 with a versioned schema.
10. Cross-check a sample of locally derived features against the supplied Zenodo high-level feature files before large-scale training.

**Important implementation rule:** if FastJet contrib or an equivalent substructure implementation cannot be installed cleanly, use the closest documented equivalent and record the exact definition difference. No silent substitution is allowed.

### 7.3 Methods compared

| ID | Method | Supervision | Role |
|---|---|---|---|
| M0 | No anomaly cut; inclusive `mJJ` spectrum | None | Baseline for bump-hunt visibility |
| M1 | MLP autoencoder, reconstruction error | Unsupervised | Standard neural baseline |
| M2 | Isolation Forest | Unsupervised | Classical baseline |
| M3 | **Deep SVDD** | Unsupervised | Main one-class method |
| M4 | **Deep SAD**, with `k` labelled signal events (`k = 10, 100, 1000`) | Semi-supervised | Main few-label method |
| M5 | Supervised MLP or XGBoost using the same five detector inputs | Supervised | Feature-space supervised reference |
| M6 | Deep SVDD with `mJJ` deliberately included | Unsupervised | Mass-leakage/sculpting control only; not a performance competitor |

M1, M3, M4, and M6 should use a shared MLP backbone family and comparable parameter budgets wherever practical. Exact hidden dimensions, latent dimensions, optimizer, batch size, maximum epochs, early stopping rule, and regularization are committed in configuration files before test evaluation.

M5 is called a **supervised reference**, not an absolute upper bound, because it is deliberately restricted to the same five non-resonant detector inputs as the main anomaly detectors.

Deep SVDD implementation notes follow Ruff et al. (ICML 2018): a bias-free network, hypersphere centre fixed after an initialization pass, optional autoencoder pretraining, and monitoring for collapse/degenerate embeddings. Deep SAD follows Ruff et al. (ICLR 2020). The exact objective variant and hyperparameters are committed before test evaluation.

### 7.4 Training regimes

| Regime | Training data | Signal labels available to training? | Purpose |
|---|---|---:|---|
| **R1 Oracle background-only** | Pure background train pool | No | Idealized unsupervised reference |
| **R2 Contaminated** | Background train pool + injected 2-prong signal at `f_train` | No | Robustness to realistic contamination |
| **R3 Semi-supervised** | R2 pool + `k` labelled 2-prong signal events for Deep SAD | Yes, only for those `k` events | Few-label benefit under contamination |

R1 is explicitly an **oracle** condition: the truth bit is used to construct a pure-background training pool. It is not presented as representative of a truly blind search.

### 7.5 Contamination grid

Use the core grid:

```text
f_train ∈ {0, 0.1%, 0.5%, 1.0%}
```

For Deep SAD, evaluate the interaction between contamination and label budget using a reduced but informative grid:

```text
f_train ∈ {0, 0.5%, 1.0%}
 k ∈ {10, 100, 1000}
```

The `k = 10` labelled events must be a deterministic subset of the `k = 100` set, which must itself be a deterministic subset of the `k = 1000` set. This removes an avoidable source of sampling noise from comparisons across label budgets.

The `k` explicitly labelled signal events must be **removed from the unlabeled training pool** so that the same event cannot simultaneously be treated as a labelled and unlabelled example.

All labelled-event IDs are stored in the experiment metadata.

### 7.6 Held-out 3-prong signal generalization

The Zenodo R&D dataset contains a separate 3-prong signal sample. When the implementation is ready without changing the core pipeline, train/tune the main detector on 2-prong data and evaluate on the unseen 3-prong signal.

The model must never see 3-prong signal events during training or hyperparameter selection.

Report:

- Known-signal performance on the 2-prong test set.
- Held-out-topology performance on the 3-prong test set.
- The difference in AUC and fixed-working-point background rejection between the two signal topologies.

This experiment is a generalization study, not evidence that the detector is model-independent in the broad sense.

### 7.7 Hyperparameter selection and leakage rules

- Selection uses the validation split only.
- M1–M3 unsupervised hyperparameters are selected without using signal labels.
- M4 Deep SAD may use only the declared `k` labelled signal events as part of its semi-supervised objective; no additional signal labels may be used for tuning.
- M5 is supervised and may use all training labels, but no test labels.
- M6 is configured exactly like M3 except for the deliberate inclusion of `mJJ`.
- All hyperparameter grids and selection criteria are stored in version-controlled YAML files before test evaluation.
- Any threshold used for a production-style cut (for example, fixed background efficiency) must be selected from training/validation information and then applied once to the test set.
- Test labels may be used to **measure** ROC/AUC/SIC after the detector and protocol are frozen, but never to select the model, hyperparameters, detector architecture, or test operating point.

### 7.8 Controlled `mJJ` leakage ablation

M6 exists only to demonstrate the effect of giving the model direct access to the resonant variable.

Compare M3 and M6 on:

- ROC AUC.
- Maximum SIC.
- Fixed-working-point background rejection.
- Score–`mJJ` dependence.
- Background `mJJ` distribution before and after anomaly-score cuts.
- Fixed-window bump-hunt behaviour.

M6 must not be used to redefine the main method, select the final architecture, or support a claim of improved anomaly detection.

## 8. Evaluation protocol (pre-registered)

Metrics and statistical procedures are fixed before looking at test results. Any remaining implementation choices must be frozen using training/validation information before the test tag is created.

### 8.1 Core detection metrics

| Metric | Definition | Reported for |
|---|---|---|
| ROC AUC | Signal-versus-background ranking metric | M1–M6 |
| Background rejection | `1 / εB` at fixed `εS ∈ {1%, 5%, 10%, 30%, 50%}` | M1–M6 |
| Significance improvement | `SIC = εS / sqrt(εB)` | M1–M6 |
| Maximum SIC | Maximum of the test ROC-derived SIC curve, reported descriptively with its threshold | M1–M6 |
| Contamination robustness | Change in AUC, rejection, and SIC versus `f_train` | M1–M4 |
| Held-out topology generalization | 2-prong versus 3-prong performance using the same frozen detector | M3–M4, conditional on execution |

The fixed signal-efficiency points are the primary working-point comparison. Maximum SIC is secondary because its threshold is obtained by scanning the evaluated sample and should therefore not be treated as a preselected operating point.

### 8.2 Mass-sculpting evaluation

Mass sculpting is evaluated for M1–M6. M0 is an inclusive spectrum reference and is not treated as a detector score.

For background events only, apply anomaly-score thresholds chosen on the validation background to target:

```text
εB = 10%
εB = 1%
```

Then report:

- Overlaid `mJJ` histograms before and after the cut.
- A normalized distribution comparison using Jensen–Shannon divergence.
- Score–`mJJ` dependence on the background sample.
- `εB(mJJ)` or an equivalent binned background acceptance diagnostic.

The mass-sculpting analysis must not use signal labels to choose the detector threshold.

### 8.3 Fixed-window bump-hunt benchmark

The benchmark uses the known 2-prong resonance centered near 3.5 TeV. The primary signal window is:

```text
3.3 TeV < mJJ < 3.7 TeV
```

This is a **fixed-window benchmark**, not a global discovery search. No global discovery significance is claimed.

The sideband analysis must predefine and freeze before test evaluation:

- Full fit range.
- Signal-region exclusion window.
- Histogram binning or unbinned treatment.
- Background functional family.
- Parameter-order/complexity rule.
- Likelihood or test-statistic definition.
- Treatment of statistical uncertainty.
- Any fit-quality requirement.
- Local p-value/significance calculation.

The default starting configuration is a smooth sideband fit with the signal window excluded. The exact fit family, complexity, range, and binning are selected using training/validation information only and frozen before the test tag.

For each applicable method, report:

1. Inclusive/pre-cut `mJJ` spectrum.
2. Post-anomaly-score-cut spectrum at `εB = 10%`.
3. Post-anomaly-score-cut spectrum at `εB = 1%`.
4. Sideband-fit background expectation in the signal window.
5. Observed count in the signal window.
6. Local significance or equivalent test statistic.
7. The change in the above relative to the inclusive spectrum.

### 8.4 Background-only null experiment

Run the complete anomaly-selection and bump-hunt procedure on a test sample containing **background only**.

The purpose is to measure whether the anomaly detector plus mass analysis can manufacture an apparent excess when no signal is present.

Report at least:

- Background `mJJ` shape before and after anomaly-score cuts.
- Fitted signal-window excess under the null.
- Local significance under the null.
- Distribution of the null statistic over bootstrap or pseudo-experiments, using a declared procedure.
- Fraction of null pseudo-experiments exceeding the chosen local-significance threshold.

Use the same fixed signal window and the same frozen statistical procedure as the signal-injected benchmark.

### 8.5 Uncertainty reporting

Keep two uncertainty sources separate:

**Training/seed variance**

- At least 5 independent random seeds for every stochastic method.
- Report mean and standard deviation across seeds.

**Finite-test-sample uncertainty**

- Use bootstrap confidence intervals on the frozen test set.
- For pairwise method comparisons, use a paired bootstrap on the same test events wherever possible.

Do not combine seed variance and test-set uncertainty into a single unexplained error bar.

### 8.6 Reporting rules

- Same frozen test events, preprocessing, feature schema, and evaluation code for all methods.
- Report all methods, including methods that perform worse than the baselines.
- Report exact sample sizes used for every metric.
- Report the test signal prevalence whenever a prevalence-dependent quantity is shown.
- All results are explicitly described as being on simulated data.
- Any deviation from this protocol goes into the `Deviations` section of the technical note before the result is interpreted.

## 9. Requirements

### 9.1 Functional

| ID | Requirement |
|---|---|
| FR-1 | `make data` downloads the selected Zenodo files, verifies checksums, records a manifest, and stores raw files outside git. |
| FR-2 | `make features` produces the cached feature table from raw events in chunks, with a logged schema version and a raw-vs-reference feature cross-check. |
| FR-3 | Training scripts for M1–M6, each driven by a YAML config and a seed. |
| FR-4 | An evaluation script produces the full metrics table, ROC/SIC curves, fixed-working-point results, sculpting plots, null-test results, and bump-hunt results from saved model outputs. |
| FR-5 | Experiment tracking (MLflow) records configs, seeds, metrics, git commit, data manifest version, and environment identifier for every run. |
| FR-6 | `make reproduce` regenerates the complete benchmark from a fresh clone using the frozen configuration. |
| FR-7 | `make reproduce-fast` regenerates a reduced validation/demo run from cached features for quick verification and CI. |
| FR-8 | Split indices, feature-schema version, configuration files, and the frozen evaluation protocol are committed and tagged before final test evaluation. |

### 9.2 Non-functional

| ID | Requirement |
|---|---|
| NFR-1 | Runs on a laptop or free Colab GPU/CPU. Peak memory remains below 8 GB during normal feature generation and training. |
| NFR-2 | Deterministic given a seed where the software stack permits it; remaining nondeterminism is documented. |
| NFR-3 | Dockerfile or pinned environment file so another machine can reproduce the environment. |
| NFR-4 | Type hints and a small test suite covering feature computation, split integrity, scaler isolation, metric functions, and threshold calibration. |
| NFR-5 | Clear README: problem, dataset, method, headline results, exact reproduction steps, and limitations. |
| NFR-6 | Full reproduction has a measured and documented runtime and disk footprint on at least one reference machine. |
| NFR-7 | Model parameter counts and key training-budget settings are logged so comparisons can be interpreted fairly. |

## 10. Deliverables

1. Public GitHub repository with code, configs, frozen splits, data manifest, and tests.
2. Results table covering all core methods and all required experiments.
3. Figures: ROC curves, fixed-working-point rejection, SIC curves, contamination sweeps, mass-sculpting plots, score–`mJJ` dependence, null-test plots, and bump-hunt spectra.
4. A 3–4 page technical note covering the method, protocol, results, null test, and limitations.
5. A CV bullet with real numbers only after results exist.
6. Optional short portfolio summary.

## 11. Timeline

Dates assume a start on Monday 28 September 2026.

| Week | Dates | Milestones | Exit check |
|---|---|---|---|
| **1** | 28 Sep – 4 Oct | Repo/environment setup. Download and verify v5 dataset. Raw chunk reader. Anti-kT clustering. Exact five-feature extraction. Feature cross-check against supplied high-level files. Freeze split generator. M0/M1/M2 baselines. Draft bump-hunt configuration and null-test pipeline. | Feature table cached, split files committed, raw/feature cross-check passes, baseline validation AUC computed, bump-hunt choices documented. |
| **2** | 5 – 11 Oct | Deep SVDD, Deep SAD, supervised reference, M6 mass-aware control. Validation-only hyperparameter search. Seed management. Contamination and label-budget matrix. | M1–M6 produce validation outputs across required seeds; configs frozen; no unresolved leakage issues. |
| **3** | 12 – 18 Oct | Contamination sweep. Sculpting analysis. Null test. Fixed-window bump hunt. Conditional 3-prong generalization. **Freeze code/configs/data manifest, tag release candidate, then run test evaluation once.** | Complete frozen test results table and figures. |
| **4** | 19 – 25 Oct | README. Technical note. Reproducibility checks. Docker/environment cleanup. CV update. Fresh-clone test. Buffer for any failed experiment. | Fresh-clone reproduction succeeds, repo public, `v1.0` tag created, claims checked against logged runs. |

### Scope rule

If week 1 slips by more than 2 days:

1. First cut the 3-prong generalization experiment.
2. Then cut M2 Isolation Forest only if necessary.
3. Do **not** cut the core contamination analysis, mass-sculpting analysis, or background-only null test.
4. Do not remove leakage controls or reproducibility requirements to save time.

## 12. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Jet clustering or substructure features take too long | Delays everything | Cluster in chunks, cache features early, use the supplied high-level files as a cross-check, and allow a documented equivalent implementation by day 3. |
| Deep SVDD hypersphere collapse or unstable training | Weak main result | Bias-free network, centre initialization, optional pretraining, collapse diagnostics, gradient/score monitoring, and honest failure reporting. |
| Detectors sculpt `mJJ` | Fake or exaggerated bump | Exclude `mJJ` from main inputs, measure score–mass dependence, run sculpting plots, and include M6 as a deliberate leakage control. |
| Bump-fit specification is unstable | Ambiguous significance | Freeze fit family/range/binning on validation data only and run alternative-fit robustness checks before the test tag. |
| Null test produces apparent excesses | Threatens the interpretation of anomaly cuts | Treat this as a result, not a failure; quantify false-bump behaviour and report it beside signal results. |
| Baselines beat Deep SVDD/SAD | Main method underperforms | Report it. Analyse feature choice, contamination sensitivity, capacity, and generalization rather than changing the protocol after seeing the test result. |
| Signal contamination breaks unsupervised training | Performance degradation | This is an explicit research axis; quantify it rather than avoiding it. |
| Deep SAD labelled events leak into unlabeled pool | Invalid semi-supervised comparison | Store labelled event IDs and assert disjointness in automated tests. |
| Scope creep | Missed deadline | Everything outside the core method/evaluation matrix is a stretch goal. |
| Overclaiming | Damages credibility | Claims policy is enforced in code/review checklist and every number must map to a logged run. |

## 13. Definition of done and CV output

The project is **done** when all of these are true:

- `make reproduce` works from a fresh clone.
- The results table covers M1–M6, with M0 as the inclusive bump-hunt reference.
- Every stochastic method has at least 5 seeds.
- Fixed-working-point metrics and maximum SIC are both reported.
- Contamination and Deep SAD label-budget experiments are complete.
- Sculpting and null-test results are reported.
- The fixed-window bump hunt is complete with a frozen fit specification.
- The 3-prong generalization experiment is either completed or explicitly recorded as dropped with a reason.
- README and technical note state limitations clearly.
- The repo is public and tagged `v1.0`.

### CV bullet template

> **Anomaly Detection on the LHC Olympics 2020 R&D Dataset** | *Python, PyTorch* | `<year>`
> – Benchmarked Deep SVDD and Deep SAD against autoencoder, Isolation Forest, and supervised references on 1.1M simulated dijet events using leakage-safe splits, controlled contamination, fixed working points, and `<N>` random seeds.
> – `<Result: real measured values only>`; quantified mass sculpting, null false-bump behaviour, and fixed-window bump-hunt changes under anomaly-score selection. Simulation-only. Code.

## 14. Claims policy

- Every number in the write-up and CV comes from a logged run in the repository.
- Results are described as **simulation benchmarks**, not physics sensitivity, exclusion limits, or discovery potential.
- R1 is described as an oracle background-only condition, not as a realistic guarantee that a real dataset is perfectly background-clean.
- Semi-supervised results are labelled as using signal labels and are not presented as model-agnostic.
- M5 is a supervised reference constrained to the same five detector inputs; it is not an absolute performance ceiling for all possible supervised models.
- M6 is a diagnostic control for direct `mJJ` leakage, not a competing final detector.
- A fixed-window local significance is not presented as a global discovery significance and no look-elsewhere correction is claimed unless a dedicated scan is actually performed.
- Negative and mixed results are included.
- The existence of a high-performing detector does not imply sensitivity to arbitrary unknown physics signals.
- Any deviation from this PRD (dropped experiments, changed features, changed fit procedure, changed seeds, changed splits) is recorded in a `Deviations` section of the technical note.

## 15. Stretch goals (only after the definition of done is met)

1. Export the `mJJ` spectrum before and after cuts to ROOT format and plot it with ROOT or `uproot`, adding ROOT tooling to the project.
2. Small C++ implementation of histogramming and the sideband fit, checked against the Python results.
3. Constituent-level input using a permutation-invariant network as an alternative to the five-feature detector input.
4. Add an explicit `mJJ`-decorrelation method and compare its effect on sculpting.
5. Apply the frozen pipeline to a challenge black-box dataset for a blind-style check.
6. Run a broader mass-window scan with an explicit trials-factor/look-elsewhere treatment.

## 16. Open questions and freeze points

| # | Question | Decide by | Freeze requirement |
|---|---|---|---|
| 1 | Exact FastJet/FastJet-contrib or equivalent implementation for `tau1`, `tau2`, `tau3` | Day 3 | Commit implementation/version and feature formulas |
| 2 | Exact contamination grid and sample construction details | End of week 1 | Commit YAML config and event-selection seed |
| 3 | Exact bump-hunt fit range, binning, background family, and statistical test | End of week 1 | Commit evaluation config before test tag |
| 4 | Exact null pseudo-experiment/ bootstrap procedure | End of week 1 | Commit evaluation config before test tag |
| 5 | Exact 3-prong preprocessing compatibility | End of week 1 | Either freeze execution or record as dropped |
| 6 | Reference machine runtime and disk footprint | End of week 2 | Record in README |
| 7 | Whether the advisor wants the write-up cross-referenced with the TF-QKD paper | Week 4 | Document the choice in the technical note |

## 17. References

1. G. Kasieczka, B. Nachman, D. Shih et al., *The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics*, arXiv:2101.08320.
2. G. Kasieczka, B. Nachman, D. Shih, *R&D Dataset for LHC Olympics 2020 Anomaly Detection Challenge*, Zenodo, v5, DOI `10.5281/zenodo.6466204`.
3. L. Ruff et al., *Deep One-Class Classification*, ICML 2018 (Deep SVDD).
4. L. Ruff et al., *Deep Semi-Supervised Anomaly Detection*, ICLR 2020 (Deep SAD).
5. M. Cacciari, G. P. Salam, G. Soyez, *FastJet User Manual*.
6. LHC Olympics 2020 public dataset documentation and supplied feature files, including the 2-prong and 3-prong R&D samples.

---

## Appendix A — Required repository structure

```text
lhco-anomaly-svdd/
├── README.md
├── PRD.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── Dockerfile
├── Makefile
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── m1_autoencoder.yaml
│   ├── m2_isolation_forest.yaml
│   ├── m3_deep_svdd.yaml
│   ├── m4_deep_sad.yaml
│   ├── m5_supervised.yaml
│   ├── m6_mass_aware.yaml
│   └── evaluation.yaml
├── data/
│   ├── README.md
│   ├── manifest.yaml
│   └── splits/
│       ├── background_train.npy
│       ├── background_val.npy
│       ├── background_test.npy
│       ├── signal2_train.npy
│       ├── signal2_val.npy
│       ├── signal2_test.npy
│       └── signal3_test.npy
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
│   ├── test_features.py
│   ├── test_splits.py
│   ├── test_scalers.py
│   ├── test_metrics.py
│   └── test_thresholds.py
├── scripts/
│   ├── download_data.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── reproduce.py
├── reports/
│   ├── figures/
│   ├── tables/
│   └── technical_note.md
└── mlruns/
```

## Appendix B — Final test-freeze checklist

Before the first final test run, all of the following must be checked automatically or manually:

- [ ] Dataset DOI and file checksums recorded.
- [ ] Split indices frozen.
- [ ] Train/validation/test disjointness verified.
- [ ] Deep SAD labelled-event IDs disjoint from unlabeled training pool.
- [ ] Feature schema version frozen.
- [ ] Preprocessing statistics come only from training data.
- [ ] All hyperparameter grids committed.
- [ ] Final model configurations committed.
- [ ] Random seeds committed.
- [ ] Fixed working-point calibration rules committed.
- [ ] `mJJ` excluded from M1–M5 inputs.
- [ ] M6 inclusion of `mJJ` explicitly isolated as a diagnostic.
- [ ] Sculpting thresholds fixed from background validation data.
- [ ] Bump-hunt window frozen.
- [ ] Bump-fit range/family/binning/statistical test frozen.
- [ ] Null-test procedure frozen.
- [ ] Evaluation scripts produce no model-selection side effects from test labels.
- [ ] Git release-candidate tag created.
- [ ] After this point, test results may be measured and analyzed, but the protocol is not altered.

## Appendix C — Minimal result table specification

The final report should contain at least one machine-readable row per:

```text
method × training_regime × contamination × label_budget × seed
```

with fields for:

```text
run_id
git_commit
data_manifest_version
feature_schema_version
method
regime
f_train
k_labels
seed
n_train_background
n_train_signal
n_val_background
n_val_signal
n_test_background
n_test_signal
roc_auc
rejection_at_1pct_signal_eff
rejection_at_5pct_signal_eff
rejection_at_10pct_signal_eff
rejection_at_30pct_signal_eff
rejection_at_50pct_signal_eff
max_sic
max_sic_threshold
mass_sculpting_jsd_at_10pct_bkg
mass_sculpting_jsd_at_1pct_bkg
score_mjj_dependence
bump_local_significance_pre_cut
bump_local_significance_at_10pct_bkg
bump_local_significance_at_1pct_bkg
null_test_statistic
```

This table is the source of truth for the technical note and CV claims.
