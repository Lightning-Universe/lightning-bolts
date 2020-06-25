import pytorch_lightning as pl
from pl_bolts.models.regression import LinearRegression, LogisticRegression
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
import numpy as np
import torch

from tests import reset_seed
import pytest


def test_linear_regression_model(tmpdir):
    reset_seed()

    # Test model with Sklearn Dataset
    # create dataset
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=3, random_state=1234)
    loaders = SklearnDataModule(X, y)

    model = LinearRegression(input_dim=3, learning_rate=0.01)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, loaders.train_dataloader(batch_size=2), loaders.val_dataloader(batch_size=2))
    trainer.test(model, loaders.test_dataloader(batch_size=2))
    #TODO: check loss/accuracy value


def test_logistic_regression_model(tmpdir):
    reset_seed()

    # create dataset
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=1234)
    loaders = SklearnDataModule(X, y)

    model = LogisticRegression(input_dim=10, num_classes=2, learning_rate=0.01)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, loaders.train_dataloader(batch_size=2), loaders.val_dataloader(batch_size=2))
    trainer.test(model, loaders.test_dataloader(batch_size=2))

    #TODO: check loss/accuracy value


test_logistic_regression_model('')
