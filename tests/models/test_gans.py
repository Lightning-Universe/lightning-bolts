import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from pl_bolts.models.gans import GAN


def test_gan(tmpdir):
    seed_everything()

    model = GAN(data_dir=tmpdir)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model)
    trainer.test(model)
