.PHONY: install lint typecheck test test-fast cov clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check gyroscope tests
	$(PY) -m ruff format --check gyroscope tests

fmt:
	$(PY) -m ruff format gyroscope tests
	$(PY) -m ruff check --fix gyroscope tests

typecheck:
	$(PY) -m mypy gyroscope

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -x --ff

cov:
	$(PY) -m pytest --cov=gyroscope --cov-report=term-missing

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -exec rm -rf {} +
