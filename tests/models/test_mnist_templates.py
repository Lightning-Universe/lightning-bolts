import pytorch_lightning as pl
from argparse import Namespace

from pl_bolts.models import LitMNISTModel
from tests import reset_seed


def test_mnist(tmpdir):
    reset_seed()

    model = LitMNISTModel()
    trainer = pl.Trainer(train_percent_check=0.01, val_percent_check=0.01, max_epochs=1,
                         test_percent_check=0.01, default_root_dir=tmpdir)
    trainer.fit(model)
    trainer.test()
    loss = trainer.callback_metrics['loss']

    assert loss <= 2.0, 'mnist failed'
