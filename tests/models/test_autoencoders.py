import pytorch_lightning as pl

from pl_bolts.models.autoencoders import BasicVAE, BasicAE
from tests import reset_seed


def test_vae(tmpdir):
    reset_seed()

    model = BasicVAE()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.test()
    loss = trainer.callback_metrics['loss']

    assert loss > 0, 'VAE failed'


def test_ae(tmpdir):
    reset_seed()

    model = BasicAE()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.test()
