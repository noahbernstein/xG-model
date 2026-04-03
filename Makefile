.PHONY: setup data features train evaluate all clean docker-build

setup:
	python -m venv venv && . venv/bin/activate && pip install -r requirements.txt

data:
	python -m src.data_collection.collect_shots

features:
	python -m src.features.build_features

train:
	python -m src.models.train

evaluate:
	python -m src.evaluation.evaluate

all: data features train evaluate

clean:
	rm -rf data/processed/*

docker-build:
	docker build -t xg-model .

test:
	pytest tests/ -v
