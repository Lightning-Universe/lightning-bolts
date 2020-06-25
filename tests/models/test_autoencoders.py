import pytorch_lightning as pl

from pl_bolts.models.autoencoders import VAE, AE
from tests import reset_seed


def test_vae(tmpdir):
    reset_seed()

    model = VAE(data_dir=tmpdir, batch_size=2)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model)
    trainer.test(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0, 'VAE failed'


def test_ae(tmpdir):
    reset_seed()

    model = AE(data_dir=tmpdir, batch_size=2)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model)
    trainer.test(model)
