.DEFAULT_GOAL := install
.PHONY: help print-env install install-dev tests test_import release clean format lint docs docs-html clean-docs

define PROJECT_HELP_MSG

Usage:\n
	\n
    make help\t\t\t             show this message\n
	\n
	-------------------------------------------------------------------------\n
	\t\tInstallation\n
	-------------------------------------------------------------------------\n
	make\t\t\t\t                Install fastar in the current environment\n
	\n
	-------------------------------------------------------------------------\n
	\t\tDevelopment\n
	-------------------------------------------------------------------------\n
	make install-dev\t\t 		Install fastar for development purpose\n
	make tests\t\t\t            Run units and integration tests\n
	\n
	make docs\t\t\t				Generate the documentation\n
	\n
	make release\t\t\t 			Build the distribution files\n
	\n
	make clean\t\t\t		Clean .pyc files and __pycache__ directories\n
	\n
	make envclean\t\t\t		Remove the local development environment\n
	\n
	-------------------------------------------------------------------------\n
	\t\tOthers\n
	-------------------------------------------------------------------------\n
	make lint\t\t\t			Lint\n

endef
export PROJECT_HELP_MSG

help:
	echo $$PROJECT_HELP_MSG

prepare-dev:
	curl -LsSf https://astral.sh/uv/install.sh | sh

print-env:
	uv run python --version
	uv run pip --version
	uv run pip freeze


install:
	pip install .

install-dev: prepare-dev
	uv lock && uv sync --all-groups
	uv run pre-commit install

test_import:
	uv run python -c "import fastar; print(fastar.__version__)"

release:
	uv build

tests:
	uv run pytest

clean: clean-docs
	rm -rf build/
	rm -rf fastar.egg-info/
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	rm -rf build/
	rm -rf dist/

format:
	uv run ruff format fastar

lint:
	uv run ruff check fastar

docs: clean-docs docs-html docs-pdf

docs-html: clean-html
	uv run --with jupyter sphinx-build -W --keep-going -b html docs/ _build/

docs-pdf: clean-pdf
	uv run --with jupyter sphinx-build -W --keep-going -b latex docs/ _build_pdf/
	make -C _build_pdf/

clean-html:
	rm -rf _build/

clean-pdf:
	rm -rf _build_pdf/

clean-docs: clean-html clean-pdf
	rm -rf docs/_autoapi

envclean:
	rm -r .venv
