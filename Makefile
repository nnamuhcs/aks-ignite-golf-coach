.PHONY: setup backend frontend test lint build up down

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r backend/requirements-dev.txt
	cd frontend && npm install

backend:
	cd backend && ../.venv/bin/uvicorn app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && ../.venv/bin/pytest -q

lint:
	cd backend && ../.venv/bin/ruff check app tests
	cd frontend && npm run lint

build:
	cd frontend && npm run build

up:
	docker compose up --build

down:
	docker compose down
