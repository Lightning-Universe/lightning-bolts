# PyTorchLightning Bolts

[![PyPI Status](https://badge.fury.io/py/pytorch-lightning-bolts.svg)](https://badge.fury.io/py/pytorch-lightning-bolts)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning-bolts)](https://pepy.tech/project/pytorch-lightning-bolts)
[![codecov](https://codecov.io/gh/PyTorchLightning/pytorch-lightning-bolts/branch/master/graph/badge.svg)](https://codecov.io/gh/PyTorchLightning/pytorch-lightning-bolts)

[![Documentation Status](https://readthedocs.org/projects/pytorch-lightning-bolts/badge/?version=latest)](https://pytorch-lightning-bolts.readthedocs.io/en/latest/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)
[![Next Release](https://img.shields.io/badge/Next%20Release-June%2020-purple.svg)](https://shields.io/)


## Continuous Integration
<center>

| System / PyTorch ver. | 1.4 (min. req.) | 1.5 (latest) |
| :---: | :---: | :---: |
| Linux py3.6 / py3.7 / py3.8 | ![CI testing](https://github.com/PyTorchLightning/pytorch-lightning-bolts/workflows/CI%20testing/badge.svg?branch=master) | ![CI testing](https://github.com/PyTorchLightning/pytorch-lightning-bolts/workflows/CI%20testing/badge.svg?branch=master) |
| OSX py3.6 / py3.7 / py3.8 | ![CI testing](https://github.com/PyTorchLightning/pytorch-lightning-bolts/workflows/CI%20testing/badge.svg?branch=master) | ![CI testing](https://github.com/PyTorchLightning/pytorch-lightning-bolts/workflows/CI%20testing/badge.svg?branch=master) |
| Windows py3.6 / py3.7 / py3.8 | wip | wip |

</center>

## Install
```pip install pytorch-lightning-bolts```

### From source

1. Download the project: `git clone https://github.com/PyTorchLightning/lightning-Covid19.git`

2. Setup your development environment:

**From Python Virtual Environments**

    python3.6 -m venv venv  # from repo root, use python3.6+
    source venv/bin/activate
    pip install -r requirements

**From conda**

To install all the needed dependencies, or update the conda environment: `./setup_dev_env.sh`

Then to activate the conda environment: `source path.bash.inc`

## What is it?
Bolts is a collection of useful models and templates to bootstrap your DL research even faster.
It's designed to work  with PyTorch Lightning

## Example
```python

from pl_bolts.models.autoencoders import BasicVAE
from pl_bolts.models.gans import BasicGAN
from pytorch_lightning import Trainer

vae = BasicVAE()
gan = BasicGAN()

# train VAE
vae_trainer = Trainer()
vae_trainer.fit(vae)

# train GAN
gan_trainer = Trainer()
gan_trainer.fit(gan)
```

## How to add a model
This repository is meant for model contributions from the community.
To add a model, you can start with the MNIST template (or any other model in the repo).

Please organize the functions of your lightning module in this order.

```python
import pytorch_lightning as pl

class MyModule(pl.LightningModule):
    
    # model
    def __init__(self):
    
    # computations
    def forward(self, x):
    
    # training loop
    def training_step(self, batch, batch_idx):
    
    # validation loop
    def validation_step(self, batch, batch_idx):
    def validation_end(self, outputs):
     
    # test loop
    def test_step(self, batch, batch_idx):
    def test_epoch_end(self, outputs):
    
    # optimizer
    def configure_optimizers(self):
    
    # data
    def prepare_data(self):
    def train_dataloader(self):
    def val_dataloader(self):
    def test_dataloader(self):
```
