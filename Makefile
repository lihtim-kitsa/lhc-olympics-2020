.PHONY: all env data features train evaluate reproduce reproduce-fast clean test

all: env data features train evaluate

env:
	pip install -e .[dev]

data:
	python scripts/download_data.py

features:
	python scripts/build_features.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

reproduce:
	make data
	make features
	make train
	make evaluate

reproduce-fast:
	make features
	make train
	make evaluate

clean:
	rm -rf data/raw/*
	rm -rf data/splits/*
	rm -rf mlruns/*
	rm -rf reports/figures/*
	rm -rf reports/tables/*

test:
	pytest tests/
