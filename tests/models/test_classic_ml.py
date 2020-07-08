import pytorch_lightning as pl
from pl_bolts.models.regression import LinearRegression, LogisticRegression
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataset
from pl_bolts.datamodules import MNISTDataModule
import numpy as np
from tests import reset_seed
from torch.utils.data import DataLoader


def test_linear_regression_model(tmpdir):
    reset_seed()

    # --------------------
    # numpy data
    # --------------------
    X = np.array([[1.0, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1.0, 2])) + 3
    y = y[:, np.newaxis]
    loader = DataLoader(SklearnDataset(X, y), batch_size=2)

    model = LinearRegression(input_dim=2, learning_rate=1.0)
    trainer = pl.Trainer(max_epochs=200, default_root_dir=tmpdir, progress_bar_refresh_rate=0)
    trainer.fit(
        model,
        loader,
        loader
    )

    coeffs = model.linear.weight.detach().numpy().flatten()
    assert len(coeffs) == 2
    assert coeffs[0] == 1 and coeffs[1] == 2
    trainer.test(model, loader)


def test_logistic_regression_model(tmpdir):
    pl.seed_everything(0)

    # create dataset
    dm = MNISTDataModule(num_workers=0, data_dir=tmpdir)

    model = LogisticRegression(input_dim=28 * 28, num_classes=10, learning_rate=0.001)
    model.prepare_data = dm.prepare_data
    model.train_dataloader = dm.train_dataloader
    model.val_dataloader = dm.val_dataloader
    model.test_dataloader = dm.test_dataloader

    trainer = pl.Trainer(max_epochs=3, default_root_dir=tmpdir, progress_bar_refresh_rate=0)
    trainer.fit(model)
    trainer.test(model)
    assert trainer.progress_bar_metrics['test_acc'] >= 0.9
