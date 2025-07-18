.PHONY: all all-dev print-env install install-all install-dev test_import whl uninstall clean clean-all static_code_analysis flake8 pylint apidoc clean-apidoc docs docs-dummy docs-html clean-docs
		

all: clean-all uninstall static_code_analysis install test_import whl docs
		

all-dev: clean-all uninstall static_code_analysis install-dev test_import whl docs
		

print-env:
	python --version
	pip --version
	pip freeze

install: uninstall
	pip install .

install-dev: uninstall
	pip install -e .

test_import:
	python -c "import fastar"

whl:
	python -m build

uninstall:
	pip uninstall -y fastar

clean:
	rm -rf build/
	rm -rf fastar.egg-info/
	rm -rf fastar/__pycache__/
	rm -rf fastar/*/__pycache__/
	rm -rf fastar/*/*/__pycache__/
	rm -rf build/
	rm -rf dist/

clean-all: clean
	rm -f fastar/_version.py

static_code_analysis: flake8 pylint
		

flake8:
	flake8 --max-complexity 10 --exclude _version.py fastar/ examples/

pylint:
	pylint --ignore=_version.py fastar/ examples/

apidoc: clean-apidoc
	sphinx-apidoc -H "Reference / API" -M -o docs/api/ fastar/

clean-apidoc:
	rm -rf docs/api/

docs: docs-dummy docs-html
		

docs-dummy: apidoc
	make -C docs/ dummy SPHINXOPTS="-W"

docs-html: apidoc
	make -C docs/ html

clean-docs: clean-apidoc
	rm -rf docs/_build/
