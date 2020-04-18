import tests.base.utils as tutils
import pytorch_lightning as pl
import pytest

from pytorch_lightning_bolts.models.vaes import VAE


def test_vae(tmpdir):
    """ Test that an error is thrown when no `training_step()` is defined """
    tutils.reset_seed()

    vae = VAE()
    trainer = pl.Trainer(train_percent_check=0.01, val_percent_check=0.1, max_epochs=1)
    trainer.fit(vae)
    loss = trainer.callback_metrics['loss']

    assert loss <= 315, 'vae failed'
