.PHONY: lint test test-cov docs clean

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=election_forecasting --cov-report=html --cov-report=term --cov-report=xml -v

docs:
	uv run sphinx-build -b html docs/source docs/build/html

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

quality-check: lint test
