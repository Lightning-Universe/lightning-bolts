<<<<<<< HEAD
.PHONY: test

test:
	# use this to run tests
	rm -rf _ckpt_*
	rm -rf ./tests/save_dir*
	rm -rf ./tests/mlruns_*
	rm -rf ./tests/cometruns*
	rm -rf ./tests/wandb*
	rm -rf ./tests/tests/*
	rm -rf ./lightning_logs
	python -m coverage run --source pl_bolts -m pytest pl_bolts tests -v --flake8
	python -m coverage report -m

isort:
	isort .
=======
.PHONY: test clean

test:
	# install APEX, see https://github.com/NVIDIA/apex#linux
	# to imitate SLURM set only single node
	export SLURM_LOCALID=0

	# use this to run tests
	rm -rf _ckpt_*
	rm -rf ./lightning_logs
	python -m coverage run --source pytorch_lightning -m pytest pytorch_lightning tests pl_examples -v --flake8
	python -m coverage report -m

	# specific file
	# python -m coverage run --source pytorch_lightning -m py.test --flake8 --durations=0 -v -k

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns" )
>>>>>>> 90c1c0f68b4983c685e9d009482890e578800439
