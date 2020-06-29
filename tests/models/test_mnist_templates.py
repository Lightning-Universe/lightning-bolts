import pytorch_lightning as pl

from pl_bolts.models import LitMNIST
from tests import reset_seed


def test_mnist(tmpdir):
    reset_seed()

    model = LitMNIST(data_dir=tmpdir)
    trainer = pl.Trainer(limit_train_batches=0.01, limit_val_batches=0.01, max_epochs=1,
                         limit_test_batches=0.01, default_root_dir=tmpdir)
    trainer.fit(model)
    trainer.test(model)
    loss = trainer.callback_metrics['loss']

    assert loss <= 2.0, 'mnist failed'
