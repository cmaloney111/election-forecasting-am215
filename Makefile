.PHONY: lint test test-cov docs clean build upload upload-test profile profile-parallel profile-view profile-view-parallel

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

mypy:
	uv run mypy src/

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term --cov-report=xml -v

docs:
	uv run sphinx-build -b html docs/source docs/build/html

build:
	rm -rf dist/
	uv run python -m build

upload-test:
	uv run twine upload --repository testpypi dist/*

upload:
	uv run twine upload dist/*

profile:
	uv run python -m cProfile -o election_forecast.prof -m src.scripts.run_all_models --dates 8
	@echo "\nProfile saved to election_forecast.prof"
	@echo "View with: make profile-view"
	@echo "Also, change cutoff to 1/100"

profile-parallel:
	uv run python -m cProfile -o election_forecast_parallel.prof -m src.scripts.run_all_models --dates 8 --parallel 4
	@echo "\nProfile saved to election_forecast_parallel.prof"
	@echo "View with: make profile-view-parallel"

profile-view:
	uv run snakeviz election_forecast.prof

profile-view-parallel:
	uv run snakeviz election_forecast_parallel.prof

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.prof" -delete

quality-check: lint mypy test
