.PHONY: lint test test-cov docs clean build upload upload-test

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

build:
	rm -rf dist/
	uv run python -m build

upload-test:
	uv run twine upload --repository testpypi dist/*

upload:
	uv run twine upload dist/*

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.prof" -delete

quality-check: lint test
