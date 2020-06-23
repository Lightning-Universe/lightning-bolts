import pytorch_lightning as pl

from pl_bolts.models.gans import GAN
from tests import reset_seed


def test_gan(tmpdir):
    reset_seed()

    model = GAN(data_dir=tmpdir)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model)
    trainer.test(model)
