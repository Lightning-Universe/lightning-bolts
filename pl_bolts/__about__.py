__version__ = '0.3.3'
__author__ = 'PyTorchLightning et al.'
__author_email__ = 'name@pytorchlightning.ai'
__license__ = 'Apache-2.0'
__copyright__ = f'Copyright (c) 2020-2021, {__author__}'
__homepage__ = 'https://github.com/PyTorchLightning/lightning-bolts'
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

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__homepage__",
    "__license__",
    "__version__",
]
