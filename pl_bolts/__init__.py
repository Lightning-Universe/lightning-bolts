"""Root package crossroad."""

import os

from pl_bolts.info import (  # noqa: F401
    __author__,
    __author_email__,
    __copyright__,
    __docs__,
    __homepage__,
    __license__,
    __version__,
)

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
    'callbacks',
    'datamodules',
    'datasets',
    'losses',
    'metrics',
    'models',
    'optimizers',
    'transforms',
    'utils',
]
