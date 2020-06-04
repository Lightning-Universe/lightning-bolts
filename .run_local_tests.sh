#!/usr/bin/env bash

# use this to run tests
rm -rf _ckpt_*
rm -rf ./tests/save_dir*
rm -rf ./tests/mlruns_*
rm -rf ./tests/cometruns*
rm -rf ./tests/wandb*
rm -rf ./tests/tests/*
rm -rf ./lightning_logs
python -m coverage run --source pytorch_lightning_bolts -m py.test pytorch_lightning_bolts tests -v --flake8
python -m coverage report -m
