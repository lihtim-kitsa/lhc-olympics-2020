"""Check experiment scheduling and resume bookkeeping without fitting models."""

import copy
import json
from pathlib import Path
import shutil

import pandas as pd

from scripts import reproduce


def test_full_sweep_headline_seeds_and_resume_manifest(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reproduce.subprocess, "run", lambda *args, **kwargs: None)
    trained = []
    monkeypatch.setattr(
        reproduce, "train_model", lambda config: trained.append(copy.deepcopy(config))
    )

    def evaluate(config):
        directory = tmp_path / "reports/tables"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "results.csv"
        name, seed = config["model"]["name"], config["training"]["seed"]
        fraction, labels = config["data"]["f_train"], config["data"].get("k_labels", 0)
        record = {
            "model": name,
            "seed": seed,
            "f_train": fraction,
            "k_labels": labels,
            "n_test_background": 10,
            "test_signal_prevalence": 0.5,
            "roc_auc": 0.5,
        }
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame([record])], ignore_index=True).to_csv(
            path, index=False
        )
        models = tmp_path / "models"
        models.mkdir(exist_ok=True)
        (models / f"{name}_f{fraction:g}_k{labels}_{seed}.pt").touch()

    monkeypatch.setattr(reproduce, "evaluate_model", evaluate)
    reproduce.run_grid()
    results = pd.read_csv(tmp_path / "reports/tables/results.csv")
    assert len(trained) == 47
    assert not results.duplicated(["model", "seed", "f_train", "k_labels"]).any()
    assert len(results[results.seed == 42]) == 23
    assert set(results.seed) == {42, 43, 44, 45, 46}
    assert results.groupby("model").seed.nunique().eq(5).all()
    sad = results[(results.seed == 42) & (results.model == "M4_DeepSAD")]
    assert set(zip(sad.f_train, sad.k_labels)) == {
        (fraction, labels) for fraction in (0.0, 0.5, 1.0) for labels in (10, 100, 1000)
    }
    trained.clear()
    reproduce.run_grid()
    assert not trained
    manifest = json.loads((tmp_path / "reports/tables/run_manifest.json").read_text())
    assert len(manifest) == 47
    assert all(row["status"] == "already completed; retained" for row in manifest)
    timing = json.loads(
        (tmp_path / "reports/tables/reproduction_timing.json").read_text()
    )
    assert timing["retained_runs"] == 47
    assert timing["completed_runs"] == 0
