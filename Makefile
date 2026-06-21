.PHONY: install dev ingest api ui eval test lint up down

install:
	pip install ".[ui]"

dev:
	pip install ".[dev,eval,ui]"

ingest:
	python -m rag.ingest --corpus data/corpus

api:
	uvicorn rag.api:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/streamlit_app.py

eval:
	python eval/run_eval.py --reingest

test:
	pytest

lint:
	ruff check src eval ui

up:
	docker compose up --build

down:
	docker compose down -v
