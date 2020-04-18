# MNIST Template
Use this template to bootstrap your models.

You can use this model in two ways:

## Research use
For research, it's recommended you define the dataloaders inside
the Lightning Module.

Fit as follows

```python
import pytorch_lightning as pl
from pytorch_lightning_bolts import LitMNISTModel
from argparse import ArgumentParser

parser = ArgumentParser()
parser = LitMNISTModel.add_model_specific_args(parser)
args = parser.parse_args()

# model
model = LitMNISTModel(hparams=args)

# train
trainer = pl.Trainer()
trainer.fit(model)

# when  training completes you can  run test set
trainer.test()
```

## Production  use
If you want to use this for production, feature extractor or similar use,
then it makes sense to define the datasets outside of the LightningModule.

```python
import os
import pytorch_lightning as pl
from pytorch_lightning_bolts import LitMNISTModel
from argparse import ArgumentParser
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

parser = ArgumentParser()
parser = LitMNISTModel.add_model_specific_args(parser)
args = parser.parse_args()

# model
model = LitMNISTModel(hparams=args)

# Train / val split
train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(train_dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
val_loader = DataLoader(mnist_val, batch_size=args.batch_size)

# test split
mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

trainer = pl.Trainer()
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

# when  training completes you can  run test set
trainer.test(test_dataloaders=test_loader)

```