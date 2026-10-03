.PHONY: all setup data features splits train evaluate reproduce reproduce-fast report test

all: reproduce

setup:
	python -m pip install -e ".[dev]"

data:
	python scripts/download_data.py

features:
	python scripts/build_features.py

splits:
	python src/data/make_dataset.py

# Convenience single-configuration commands. Use `make reproduce` for the full grid.
train:
	python scripts/train.py --config configs/m1_autoencoder.yaml

evaluate:
	python scripts/evaluate.py --config configs/m1_autoencoder.yaml

reproduce:
	python scripts/reproduce.py

reproduce-fast:
	python scripts/reproduce_fast.py

report:
	python scripts/build_report.py

test:
	python -m pytest tests/

