from pl_bolts.callbacks import LatentDimInterpolator
from pl_bolts.models.gans import GAN
from pytorch_lightning.loggers.base import DummyLogger


def test_latent_dim_interpolator(tmpdir):

    class FakeTrainer(object):
        def __init__(self):
            self.current_epoch = 1
            self.global_step = 1
            self.logger = DummyLogger()

    model = GAN()
    cb = LatentDimInterpolator(interpolate_epoch_interval=2)

    cb.on_epoch_end(FakeTrainer(), model)
