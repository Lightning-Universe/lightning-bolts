"""Root package info."""

import os

__version__ = '0.1.0-dev5'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'name@pytorchlightning.ai'
__license__ = 'TBD'
__copyright__ = 'Copyright (c) 2020-2020, %s.' % __author__
__homepage__ = 'https://github.com/PyTorchLightning/pytorch-lightning-bolts'
__docs__ = "PyTorch Lightning Bolts is a community contribution for ML researchers."
__long_doc__ = """
What is it?
-----------
Bolts is a collection of useful models and templates to bootstrap your DL research even faster.
It's designed to work  with PyTorch Lightning

Example
-------

.. code-block:: python

    from pytorch_lightning_bolts.vaes import VAE
    from pytorch_lightning import Trainer

    vae = VAE()

    # train VAE
    trainer = Trainer()
    trainer.fit(vae)

How to add a model
------------------

This repository is meant for model contributions from the community.
To add a model, you can start with the MNIST template (or any other model in the repo).
Please organize the functions of your lightning module.
"""

PACKAGE_ROOT = os.path.dirname(__file__)

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __LIGHTNING_BOLT_SETUP__
except NameError:
    __LIGHTNING_BOLT_SETUP__ = False

if __LIGHTNING_BOLT_SETUP__:
    import sys  # pragma: no-cover
    sys.stdout.write(f'Partial import of `{__name__}` during the build process.\n')  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:
    from pytorch_lightning_bolts.models.mnist_template import LitMNISTModel

    __all__ = [
        'LitMNISTModel'
    ]
