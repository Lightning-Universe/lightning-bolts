import pytorch_lightning as pl
from argparse import Namespace

from pytorch_lightning_bolts import LitMNISTModel
from tests import reset_seed


def test_mnist(tmpdir):
    """ Test that an error is thrown when no `training_step()` is defined """
    reset_seed()

    params = {'hidden_dim': 128, 'batch_size': 32, 'learning_rate': 0.001}
    model = LitMNISTModel(Namespace(**params))
    trainer = pl.Trainer(train_percent_check=0.01, val_percent_check=0.01, max_epochs=1)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss <= 2.0, 'mnist failed'
