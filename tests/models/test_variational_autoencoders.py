import pytorch_lightning as pl

from pytorch_lightning.bolts.models.vaes.basic.template import BasicVAE
from tests import reset_seed


def test_vae(tmpdir):
    """ Test that an error is thrown when no `training_step()` is defined """
    reset_seed()

    vae = BasicVAE()
    trainer = pl.Trainer(
        train_percent_check=0.01,
        val_percent_check=0.1,
        max_epochs=1,
        test_percent_check=0.01
    )
    trainer.fit(vae)
    trainer.test()
    loss = trainer.callback_metrics['loss']

    assert loss <= 350, 'vae failed'
