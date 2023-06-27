sources = src scripts energizer

.PHONY: test format lint unittest coverage pre-commit clean
test: format lint unittest

format:
	isort $(sources)
	black $(sources)
	nbqa isort notebooks
	nbqa black notebooks --line-length 100

lint:
	ruff $(sources)

unittest:
	pytest

coverage:
	pytest --cov=$(sources) --cov-branch --cov-report=term-missing tests

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf .mypy_cache .pytest_cache
	rm -rf *.egg-info
	rm -rf .tox dist site
	rm -rf coverage.xml .coverage
	rm -rf */lightning_logs/
	rm -rf site
	rm -rf outputs/dry_run

serve_docs:
	mkdocs serve --watch .


clean-poetry-cache:
	rm -rf ~/.cache/pypoetry/virtualenvs/