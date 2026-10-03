"""Refresh manuscript tables and figures from the saved evaluation table."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
HEADLINES = {
    "M1_Autoencoder": ("M1: Autoencoder", 0.0, 0, "AE"),
    "M2_IsolationForest": ("M2: Isolation Forest", 0.0, 0, "IF"),
    "M3_DeepSVDD": ("M3: Deep SVDD", 0.0, 0, "SVDD"),
    "M4_DeepSAD": ("M4: Deep SAD", 0.5, 100, "SAD"),
    "M5_Supervised": ("M5: Supervised", 0.0, 1000, "Sup"),
    "M6_MassAware": ("M6: Mass-aware SVDD", 0.0, 0, "Mass"),
}


def select(data, model, fraction, labels):
    return data[
        (data.model == model) & (data.f_train == fraction) & (data.k_labels == labels)
    ].sort_values("seed")


def mean_sd(values):
    return f"{values.mean():.3f}\\pm{values.std(ddof=1):.3f}"


def replace_table(text, label, rows):
    start = text.index(r"\label{" + label + "}")
    first = text.index(r"\midrule", start) + len(r"\midrule")
    last = text.index(r"\bottomrule", first)
    return text[:first] + "\n" + "\n".join(rows) + "\n" + text[last:]


def refresh():
    data = pd.read_csv(ROOT / "reports/tables/results.csv")
    if data.duplicated(["model", "f_train", "k_labels", "seed"]).any():
        raise ValueError("Duplicate result configurations")
    selected = {}
    macros, headline_rows, transfer_rows, bump_rows = [], [], [], []
    for model, (name, fraction, labels, prefix) in HEADLINES.items():
        group = select(data, model, fraction, labels)
        if set(group.seed) != {42, 43, 44, 45, 46} or len(group) != 5:
            raise ValueError(f"{model}: five distinct headline seeds required")
        selected[model] = group
        for field, suffix in [
            ("roc_auc", "Mean"),
            ("roc_auc_3prong", "TransferMean"),
            ("mass_sculpting_jsd_at_10pct_bkg", "JSD"),
        ]:
            macros.append(
                "\\newcommand{\\"
                + prefix
                + suffix
                + "}{"
                + f"{group[field].mean():.4f}"
                + "}"
            )
        macros.append(
            "\\newcommand{\\" + prefix + "Headline}{" + mean_sd(group.roc_auc) + "}"
        )
        macros.append(
            "\\newcommand{\\"
            + prefix
            + "TransferHeadline}{"
            + mean_sd(group.roc_auc_3prong)
            + "}"
        )
        cells = [
            name,
            f"{fraction:g}\\%",
            str(labels),
            "$" + mean_sd(group.roc_auc) + "$",
        ]
        cells += [
            f"{group[field].mean():.1f}" for field in ("rej_10", "rej_30", "rej_50")
        ]
        cells += [
            "$" + mean_sd(group.max_sic) + "$",
            f"{group.mass_sculpting_jsd_at_10pct_bkg.mean():.4f}",
            f"{group.score_mjj_dependence.mean():.3f}",
        ]
        headline_rows.append(" & ".join(cells) + r" \\")
        gap = (group.roc_auc_3prong - group.roc_auc).mean()
        transfer_rows.append(
            " & ".join(
                [
                    name,
                    "$" + mean_sd(group.roc_auc) + "$",
                    "$" + mean_sd(group.roc_auc_3prong) + "$",
                    f"${gap:+.3f}$",
                ]
            )
            + r" \\"
        )
        if model in ("M4_DeepSAD", "M5_Supervised"):
            for row in group.itertuples():
                cells = [name, str(row.seed)]
                cells += [
                    (
                        f"{getattr(row, field):.2f}"
                        if np.isfinite(getattr(row, field))
                        else "invalid fit"
                    )
                    for field in (
                        "pre_cut_local_significance",
                        "at_10pct_bkg_local_significance",
                        "at_1pct_bkg_local_significance",
                    )
                ]
                bump_rows.append(" & ".join(cells) + r" \\")
    for model in (
        "M0_TauCut",
        "M7_CWoLa",
        "M8_ANODE",
        "M11_CATHODE_CVAE",
        "M12_CWoLaTrees",
    ):
        group = data[(data.model == model) & (data.seed == 42)]
        row = group.iloc[0]
        cells = [
            model.replace("_", ": "),
            f"{row.f_train:g}\\%",
            str(int(row.k_labels)),
            f"{row.roc_auc:.3f}",
        ]
        cells += [f"{row[field]:.1f}" for field in ("rej_10", "rej_30", "rej_50")]
        cells += [
            f"{row.max_sic:.2f}",
            f"{row.mass_sculpting_jsd_at_10pct_bkg:.4f}",
            f"{row.score_mjj_dependence:.3f}",
        ]
        headline_rows.append(" & ".join(cells) + r" \\")

    path = ROOT / "reports/lhco2020_paper.tex"
    text = path.read_text(encoding="utf-8")
    if r"\input{release_results.tex}" not in text:
        text = text.replace(
            r"\begin{document}",
            r"\input{release_results.tex}" + "\n" + r"\begin{document}",
            1,
        )
        replacements = {
            r"0.806 \pm 0.009": r"\IFHeadline",
            r"0.934 \pm 0.001": r"\SADHeadline",
            r"0.824 \pm 0.008": r"\SADTransferHeadline",
            "0.806": r"\IFMean",
            "0.934": r"\SADMean",
            "0.966": r"\SupMean",
            "0.824": r"\SADTransferMean",
            "0.0126": r"\SADJSD",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
    text = text.replace(
        "averaged over seeds 42, 43 and 44.", "averaged over seeds 42--46 for M1--M6."
    )
    text = text.replace(
        "three seeds for neural models where evaluated;",
        "five seeds for the six core models;",
    )
    text = text.replace(
        "averaged over the three seeds (42, 43, 44)", "reported for seed 42"
    )
    text = text.replace("mean $\\pm$ std over seeds 42--44", "seed 42")
    text = text.replace("mean $\\pm$ s.d. over seeds 42--44", "five seeds 42--46")
    # The contamination figure is the full single-seed sweep, not a five-seed mean.
    text = text.replace("models (five seeds 42--46).", "models (seed 42).")
    text = text.replace(
        "with the lowest validation objective is selected. Three seeds\n"
        "(42, 43 and 44) are used for all headline comparisons.",
        "with the lowest validation objective is selected. Five seeds\n"
        "(42--46) are used for the six core headline comparisons;\n"
        "supplemental baselines are single-seed exploratory evaluations.",
    )
    text = replace_table(text, "tab:headline", headline_rows)
    text = replace_table(text, "tab:generalisation", transfer_rows)
    text = replace_table(text, "tab:bumphunt", bump_rows)
    sad = data[(data.model == "M4_DeepSAD") & (data.seed == 42)]
    rows = []
    for fraction in (0.0, 0.5, 1.0):
        cells = [f"{fraction:g}\\%"]
        for labels in (10, 100, 1000):
            cells.append(
                f'{select(sad, "M4_DeepSAD", fraction, labels).roc_auc.iloc[0]:.3f}'
            )
        rows.append(" & ".join(cells) + r" \\")
    text = replace_table(text, "tab:deepsad_grid", rows)
    rows = []
    for fraction in (0.0, 0.1, 0.5, 1.0):
        values = [
            data[(data.model == model) & (data.seed == 42) & (data.f_train == fraction)]
            .iloc[0]
            .roc_auc
            for model in ("M1_Autoencoder", "M2_IsolationForest", "M3_DeepSVDD")
        ]
        rows.append(
            " & ".join([f"{fraction:g}\\%"] + [f"{value:.3f}" for value in values])
            + r" \\"
        )
    text = replace_table(text, "tab:contamination", rows)
    # Drop stale numerical claims that are not established by the final diagnostics.
    text = text.replace(
        "local injected-signal significance above\n$8\\sigma$ after a 1\\% background cut.",
        "a descriptive local fixed-window diagnostic.",
    )
    text = text.replace("recovering about $80\\%$ of", "recovering much of")
    path.write_text(text, encoding="utf-8")
    (ROOT / "reports/release_results.tex").write_text(
        "% Generated from reports/tables/results.csv; five core headline seeds.\n"
        + "\n".join(macros)
        + "\n",
        encoding="utf-8",
    )
    plots(data, selected)
    print("Updated five manuscript tables and data-derived figures.")


def plots(data, selected):
    directory = ROOT / "reports/figures"
    frozen = ROOT / "reports/tables/frozen_test_scores.npz"
    if frozen.exists():
        with np.load(frozen) as samples:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for model in (
                "M1_Autoencoder",
                "M2_IsolationForest",
                "M4_DeepSAD",
                "M5_Supervised",
            ):
                fpr, tpr, _ = roc_curve(samples["labels"], samples[model])
                valid = fpr > 0
                axes[0].plot(tpr[valid], 1 / fpr[valid], label=model)
                axes[1].plot(tpr[valid], tpr[valid] / np.sqrt(fpr[valid]), label=model)
            axes[0].set(
                yscale="log", ylabel="Background rejection", title="Seed-42 ROC"
            )
            axes[1].set(ylabel="SIC", title="Seed-42 significance improvement")
            for ax in axes:
                ax.set(xlabel="Signal efficiency")
                ax.legend(fontsize=7)
            save(fig, directory / "headline_roc_sic_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for model in ("M1_Autoencoder", "M2_IsolationForest", "M3_DeepSVDD"):
        group = data[(data.model == model) & (data.seed == 42)].sort_values("f_train")
        ax.plot(group.f_train, group.roc_auc, "o-", label=model)
    ax.set(xlabel="Training signal injection (%)", ylabel="ROC AUC", ylim=(0, 1))
    ax.legend(fontsize=8)
    save(fig, directory / "contamination_auc_effect.png")
    pivot = data[(data.model == "M4_DeepSAD") & (data.seed == 42)].pivot(
        index="f_train", columns="k_labels", values="roc_auc"
    )
    fig, ax = plt.subplots(figsize=(6, 3.6))
    image = ax.imshow(pivot, vmin=0.4, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(3), [str(k) for k in pivot.columns])
    ax.set_yticks(range(3), [f"{f:g}%" for f in pivot.index])
    ax.set(xlabel="Labeled signal events", ylabel="Training signal injection")
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, f"{pivot.iloc[i, j]:.3f}", color="white", ha="center", va="center"
            )
    fig.colorbar(image, ax=ax, label="ROC AUC")
    save(fig, directory / "deepsad_auc_grid.png")
    for filename, xfield, yfield, xlabel, ylabel in [
        (
            "generalisation_scatter.png",
            "roc_auc",
            "roc_auc_3prong",
            "Two-prong ROC AUC",
            "Held-out three-prong ROC AUC",
        ),
        (
            "jsd_vs_auc_tradeoff.png",
            "mass_sculpting_jsd_at_10pct_bkg",
            "roc_auc",
            "Background mass JSD at 10% cut",
            "Two-prong ROC AUC",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for model, group in selected.items():
            ax.scatter(group[xfield], group[yfield], label=model, alpha=0.7, s=20)
            ax.scatter(group[xfield].mean(), group[yfield].mean(), marker="*", s=110)
        if xfield == "roc_auc":
            ax.plot([0.4, 1], [0.4, 1], "--", color="gray", linewidth=0.8)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=7, loc="best")
        save(fig, directory / filename)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    refresh()
