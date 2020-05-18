# PyTorchLightning Bolts







## Install
```pip install pytorch-lightning-bolts```

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
