import pytorch_lightning as pl

from pl_bolts.models.gans import BasicGAN
from tests import reset_seed


def test_gan(tmpdir):
    reset_seed()

    model = BasicGAN()
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(model)
    trainer.test()
