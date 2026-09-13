.PHONY: install lint test train export api ui up down

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

test:
	pytest

train:
	python ml/train.py --model all

export:
	python ml/export.py

api:
	uvicorn api.main:app --reload

ui:
	streamlit run ui/app.py

up:
	docker compose up --build

down:
	docker compose down
