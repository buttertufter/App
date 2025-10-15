.PHONY: help install dev test lint format clean run-ui run-api prefetch pipeline

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make dev        - Run Streamlit UI in development mode"
	@echo "  make run-ui     - Run Streamlit UI"
	@echo "  make run-api    - Run FastAPI server"
	@echo "  make test       - Run all tests"
	@echo "  make lint       - Run linters (ruff)"
	@echo "  make format     - Format code (black, ruff)"
	@echo "  make typecheck  - Run type checker (mypy)"
	@echo "  make prefetch   - Prefetch data for offline use"
	@echo "  make pipeline   - Run data pipeline"
	@echo "  make clean      - Remove cache and generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Run Streamlit UI
dev: run-ui

run-ui:
	streamlit run app/main.py

# Run FastAPI server
run-api:
	uvicorn scripts.api_server:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

# Run linter
lint:
	ruff check .

# Format code
format:
	black .
	ruff check --fix .

# Type checking
typecheck:
	mypy --ignore-missing-imports core/ data/ equations/ ui/ scripts/

# Prefetch data
prefetch:
	python -m scripts.prefetch

# Run data pipeline
pipeline:
	python -m scripts.pipeline

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Deep clean (includes cache and data)
clean-all: clean
	rm -rf cache/*
	@echo "Note: data_store/ preserved. Remove manually if needed."
