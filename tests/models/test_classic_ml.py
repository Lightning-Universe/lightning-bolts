import functools
import operator

import numpy as np
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataset
from pl_bolts.models.regression import LinearRegression, LogisticRegression


def test_linear_regression_model(tmpdir):
    seed_everything()

    # --------------------
    # numpy data
    # --------------------
    X = np.array([[1.0, 1], [1, 2], [2, 2], [2, 3], [3, 3], [3, 4], [4, 4], [4, 5]])
    y = np.dot(X, np.array([1.0, 2])) + 3
    y = y[:, np.newaxis]
    loader = DataLoader(SklearnDataset(X, y), batch_size=2)

    model = LinearRegression(input_dim=2, learning_rate=0.6)
    trainer = Trainer(
        max_epochs=400,
        default_root_dir=tmpdir,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(
        model,
        loader,
        loader,
    )

    coeffs = model.linear.weight.detach().numpy().flatten()
    np.testing.assert_allclose(coeffs, [1, 2], rtol=1e-3)
    trainer.test(model, loader)


def test_logistic_regression_model(tmpdir, datadir):
    seed_everything(0)

    # create dataset
    dm = MNISTDataModule(num_workers=0, data_dir=datadir)

    model = LogisticRegression(
        input_dim=functools.reduce(operator.mul, dm.dims, 1), num_classes=10, learning_rate=0.001
    )
    model.prepare_data = dm.prepare_data
    model.setup = dm.setup
    model.train_dataloader = dm.train_dataloader
    model.val_dataloader = dm.val_dataloader
    model.test_dataloader = dm.test_dataloader

    trainer = Trainer(
        max_epochs=3,
        default_root_dir=tmpdir,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model)
    trainer.test(model)
    # todo: update model and add healthy check
    # assert trainer.progress_bar_dict['test_acc'] >= 0.9
