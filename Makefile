.PHONY: test clean docs env

# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=1

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "lightning_log")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/generated
	rm -rf ./docs/source/*/generated
	rm -rf ./docs/source/api
	rm -rf ./_datasets

test: clean env

	# run tests with coverage
	python -m coverage run --source pl_bolts -m pytest pl_bolts tests -v
	python -m coverage report

doctest: clean env
	pip install --quiet -r docs/requirements.txt
	SPHINX_MOCK_REQUIREMENTS=0 make -C docs doctest

docs: clean
	pip install --quiet -r docs/requirements.txt
	python -m sphinx -b html -W docs/source docs/build

env:
	pip install -q -r requirements/devel.txt

format:
	isort .
	black .
	flake8 .
