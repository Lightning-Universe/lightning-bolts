"""Root package info."""

import os

__version__ = '0.3.0'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'name@pytorchlightning.ai'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2020-2021, {__author__}'
__homepage__ = 'https://github.com/PyTorchLightning/pytorch-lightning-bolts'
__docs__ = "PyTorch Lightning Bolts is a community contribution for ML researchers."
__long_doc__ = """
What is it?
-----------
Bolts is a collection of useful models and templates to bootstrap your DL research even faster.
It's designed to work  with PyTorch Lightning

Subclass Example
----------------
Use `pl_bolts` models to remove boilerplate for common approaches and architectures.
Because it uses LightningModules under the hood, you just need to overwrite
the relevant parts to your research.

How to add a model
------------------
This repository is meant for model contributions from the community.
To add a model, you can start with the MNIST template (or any other model in the repo).
Please organize the functions of your lightning module.
"""

_PACKAGE_ROOT = os.path.dirname(__file__)
_HTTPS_AWS_HUB = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"

try:
    # This variable is injected in the __builtins__ by the build process.
    # It used to enable importing subpackages when the binaries are not built.
    _ = None if __LIGHTNING_BOLT_SETUP__ else None
except NameError:
    __LIGHTNING_BOLT_SETUP__: bool = False

if __LIGHTNING_BOLT_SETUP__:  # pragma: no cover
    import sys

    sys.stdout.write(f'Partial import of `{__name__}` during the build process.\n')
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:
    from pl_bolts import callbacks, datamodules, datasets, losses, metrics, models, optimizers, transforms, utils

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
