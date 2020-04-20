"""Root package info."""

import os

__version__ = '0.1.0-dev5'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'will@pytorchlightning.ai'
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
