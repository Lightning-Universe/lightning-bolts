"""Root package crossroad."""

import os

import numpy

from pl_bolts.__about__ import *  # noqa: F401, F403

# adding compatibility for numpy >= 1.24
for tp_name, tp_ins in [("object", object), ("bool", bool), ("int", int), ("float", float)]:
    if not hasattr(numpy, tp_name):
        setattr(numpy, tp_name, tp_ins)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
_HTTPS_AWS_HUB = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"

from pl_bolts import (  # noqa: E402
    callbacks,
    datamodules,
    datasets,
    losses,
    metrics,
    models,
    optimizers,
    transforms,
    utils,
)

__all__ = [
    "callbacks",
    "datamodules",
    "datasets",
    "losses",
    "metrics",
    "models",
    "optimizers",
    "transforms",
    "utils",
]
