import functools
import operator

import pytorch_lightning as pl
from pl_bolts import datamodules
from pl_bolts.models import regression


def test_logistic_regression_model(datadir):
    pl.seed_everything(0)

    dm = datamodules.MNISTDataModule(datadir)

    model = regression.LogisticRegression(
        input_dim=functools.reduce(operator.mul, dm.dims, 1), num_classes=10, learning_rate=0.001
    )

    trainer = pl.Trainer(max_epochs=3, logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    assert trainer.state.finished
    assert trainer.callback_metrics["test_acc"] > 0.9
    assert trainer.callback_metrics["test_loss"] < 0.3
