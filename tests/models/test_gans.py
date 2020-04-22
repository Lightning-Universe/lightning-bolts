import pytorch_lightning as pl

from pl_bolts.models.gans import BasicGAN
from tests import reset_seed


def test_gan(tmpdir):
    reset_seed()

    model = BasicGAN()
    trainer = pl.Trainer(train_percent_check=0.01, max_epochs=1)
    trainer.fit(model)
