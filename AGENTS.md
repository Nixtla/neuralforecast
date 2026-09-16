# Repository Guidelines

## Project Structure & Module Organization

`neuralforecast/` contains the Python forecasting library: `core.py` provides orchestration, `tsdataset.py` handles datasets, `auto.py` defines tuning wrappers, and `common/` and `losses/` hold shared infrastructure. Forecasting models live in `neuralforecast/models/`; each exported model must be defined in its own dedicated module and exposed through `models/__init__.py`.

`tests/` contains pytest suites, with model, shared-component, and loss tests in `test_models/`, `test_common/`, and `test_losses/`. Shared fixtures live in `tests/conftest.py`. `nbs/` holds documentation notebooks; `docs/` contains documentation tooling and Mintlify content. `experiments/` holds benchmarks, and `scripts/` contains maintenance utilities.

## Build, Test, and Development Commands

- `uv venv --python 3.11`: create the recommended development environment.
- `uv sync --group dev --torch-backend auto`: install the library and development dependencies; use `--torch-backend cpu` for CPU-only development.
- `uv run pre-commit install`: enable commit hooks.
- `uv run pre-commit run --all-files`: run configured Ruff, mypy, and requirements-file checks.
- `uv run pytest`: run the suite with configured coverage reporting.
- `uv run pytest tests/test_model_file_policy.py --no-cov`: run a focused architecture check without the global coverage gate.
- `make all_docs` / `make preview_docs`: generate / preview documentation after installing the documented Quarto and Mintlify prerequisites.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions and modules, and `PascalCase` for classes. Preserve surrounding style and avoid unrelated whitespace or formatting changes. Ruff targets Python 3.10 with an 88-character line length and Pyflakes (`F`) rules; Black is available in development dependencies. Write Google-style docstrings for public APIs, which feed generated documentation.

## Testing Guidelines

Name files `test_*.py` and test functions `test_*`. Add regression tests that fail before a bug fix and pass afterward. Reuse existing fixtures and place tests beside the relevant component suite. The default pytest configuration requires 80% coverage and produces terminal and HTML reports. Focused runs may use `--no-cov`; run the relevant broader suites before submitting.

## Commit & Pull Request Guidelines

Recent history uses descriptive subjects including `feat:`, `fix:`, and `refactor(models):`; follow that pattern where appropriate. Keep PRs focused on one concern. Describe the problem, solution, and validation, link relevant issues, and include regression tests. Separate style-only changes from functional changes. Follow `CONTRIBUTING.md` for submission details; clear notebook outputs when contributing examples.
