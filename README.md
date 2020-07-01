<div align="center">

![Logo](https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/master/docs/source/_images/logos/lightning_logo.svg)

# PyTorch Lightning Bolts    

**Pretrained SOTA Deep Learning models, callbacks and more for research and production with PyTorch Lightning and PyTorch**

[![PyPI Status](https://badge.fury.io/py/pytorch-lightning-bolts.svg)](https://badge.fury.io/py/pytorch-lightning-bolts)
[![PyPI Status](https://pepy.tech/badge/pytorch-lightning-bolts)](https://pepy.tech/project/pytorch-lightning-bolts)
[![codecov](https://codecov.io/gh/PyTorchLightning/pytorch-lightning-bolts/branch/master/graph/badge.svg)](https://codecov.io/gh/PyTorchLightning/pytorch-lightning-bolts)

[![Documentation Status](https://readthedocs.org/projects/pytorch-lightning-bolts/badge/?version=latest)](https://pytorch-lightning-bolts.readthedocs.io/en/latest/)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/PytorchLightning/pytorch-lightning/blob/master/LICENSE)
[![Next Release](https://img.shields.io/badge/Next%20Release-June%2020-purple.svg)](https://shields.io/)

</div>

---   
## Trending contributors

[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/0)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/0)[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/1)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/1)[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/2)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/2)[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/3)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/3)[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/4)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/4)[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/5)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/5)[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/6)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/6)[![](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/images/7)](https://sourcerer.io/fame/williamFalcon/pytorchlightning/pytorch-lightning-bolts/links/7)


## Install
```pip install pytorch-lightning-bolts```

## Docs
- [master](https://pytorch-lightning-bolts.readthedocs.io/en/latest)
- [stable](https://pytorch-lightning-bolts.readthedocs.io/en/stable)
- [0.1.0](https://pytorch-lightning-bolts.readthedocs.io/en/0.1.0/)

## What is Bolts
Bolts is a Deep learning research and production toolbox of:

- SOTA pretrained models.
- Model components.
- Callbacks.
- Losses.
- Datasets.

## Main Goals of Bolts
The main goal of Bolts is to enable rapid model idea iteration.

#### Example 1: Finetuning on data

```python
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform
import pytorch_lightning as pl

# data
train_data = DataLoader(MyDataset(transforms=SimCLRTrainDataTransform(input_height=32)))
val_data = DataLoader(MyDataset(transforms=SimCLREvalDataTransform(input_height=32)))

# model
model = SimCLR(pretrained='imagenet2012')

# train!
trainer = pl.Trainer(gpus=8)
trainer.fit(model, train_data, val_data)
```

#### Example 2: Subclass and ideate

```python
from pl_bolts.models import ImageGPT
from pl_bolts.self_supervised import SimCLR

class VideoGPT(ImageGPT):

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = _shape_input(x)

        logits = self.gpt(x)
        simclr_features = self.simclr(x)

        # -----------------
        # do something new with GPT logits + simclr_features
        # -----------------

        loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1).long())

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}
```

## Who is Bolts for?
- Corporate production teams
- Professional researchers
- Ph.D. students
- Linear + Logistic regression heroes

## I don't need deep learning
Great! 
We have LinearRegression and LogisticRegression implementations with numpy and sklearn bridges for datasets!
But our implementations work on multiple GPUs, TPUs and scale dramatically...

[Check out our Linear Regression on TPU demo](https://colab.research.google.com/drive/13glsKiwMu1-H24cBLYaWdJ4_TxC2Z3ox?usp=sharing)

```python
from pl_bolts.models.regression import LinearRegression
from pl_bolts.datamodules import SklearnDataModule

# sklearn dataset
X, y = load_boston(return_X_y=True)
loaders = SklearnDataModule(X, y)

model = LinearRegression(input_dim=13)
trainer = pl.Trainer(num_tpu_cores=1)
trainer.fit(model, loaders.train_dataloader(), loaders.val_dataloader())
trainer.test(test_dataloaders=loaders.test_dataloader())
```

## Is this another model zoo?
No! 

Bolts is unique because models are implemented using PyTorch Lightning and structured so that they can be easily
subclassed and iterated on.

For example, you can override the elbo loss of a VAE, or the generator_step of a GAN to quickly try out a new idea.
The best part is that all the models are benchmarked so you won't waste time trying to "reproduce" or find the bugs
with your implementation.

## Team
Bolts is supported by the PyTorch Lightning team and the PyTorch Lightning community!

