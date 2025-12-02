# CI/CD Documentation

This project uses GitHub Actions for continuous integration and deployment.

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

Runs on:
- Pull requests to `main` or `dev`
- Direct pushes to `main` or `dev`

**Jobs:**

1. **Lint & Format Check**
   - Runs `ruff check` for linting
   - Runs `ruff format --check` for formatting verification

2. **Type Check (mypy)**
   - Runs `mypy src/` for static type checking

3. **Test Suite**
   - Tests across Python 3.9, 3.10, 3.11, 3.12
   - Ensures compatibility across Python versions

4. **Test Coverage**
   - Runs tests with coverage reporting
   - Uploads coverage to Codecov
   - Requires `CODECOV_TOKEN` secret

5. **Build Distribution**
   - Builds source and wheel distributions
   - Validates with `twine check`
   - Uploads build artifacts

6. **Docker Build Test**
   - Builds Docker image
   - Tests the built image
   - Uses GitHub Actions cache for faster builds

### 2. Publish Workflow (`.github/workflows/publish.yml`)

**Publish to PyPI** (runs on `main` branch or version tags):
- Triggered by:
  - Pushes to `main` branch
  - Version tags (`v*`)
  - GitHub releases
  - Manual workflow dispatch
- Uses trusted publishing (no API token needed)
- Requires PyPI trusted publisher configuration

**Publish to TestPyPI** (runs on `dev` branch):
- Triggered by pushes to `dev` branch
- Uses trusted publishing
- Good for testing releases before production

## Setup Requirements

### 1. Codecov Token
Add `CODECOV_TOKEN` to repository secrets:
1. Go to https://codecov.io/
2. Link your repository
3. Copy the upload token
4. Add to GitHub: Settings → Secrets → Actions → New repository secret
   - Name: `CODECOV_TOKEN`
   - Value: [your token]

### 2. PyPI Trusted Publishing

For **PyPI** (main):
1. Go to https://pypi.org/manage/account/publishing/
2. Add publisher:
   - Owner: `cmaloney111`
   - Repository: `election-forecasting-am215`
   - Workflow: `publish.yml`
   - Environment: `pypi`

For **TestPyPI** (dev):
1. Go to https://test.pypi.org/manage/account/publishing/
2. Add publisher with same settings but environment: `testpypi`

### 3. GitHub Environments

Create two environments in repository settings:

**pypi:**
- Protection rules: Require reviewers (optional)
- Deployment branches: Only `main` and tags

**testpypi:**
- Protection rules: None needed
- Deployment branches: Only `dev`

## Badges

The README includes status badges:
- CI status: Shows if tests are passing
- Codecov: Shows code coverage percentage
- PyPI version: Shows latest published version
- Python versions: Shows supported Python versions

## Local Development

All CI checks can be run locally:

```bash
# Run full quality check (lint, mypy, tests)
make quality-check

# Run specific checks
make lint
make mypy
make test
make test-cov

# Build distribution
make build

# Docker build
docker build -t election-forecasting .
```

## Workflow Triggers Summary

| Workflow | main | dev | PR | Release | Manual |
|----------|------|-----|----|---------| -------|
| CI       | ✓    | ✓   | ✓  |         | ✓      |
| PyPI     | ✓    |     |    | ✓       | ✓      |
| TestPyPI |      | ✓   |    |         | ✓      |
