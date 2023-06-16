from pl_bolts.callbacks import LatentDimInterpolator
from pl_bolts.models.gans import GAN

try:
    from pytorch_lightning.loggers.logger import DummyLogger  # PL v1.9+
except ModuleNotFoundError:
    from pytorch_lightning.loggers.base import DummyLogger  # PL v1.8


def test_latent_dim_interpolator():
    class FakeTrainer:
        def __init__(self) -> None:
            self.current_epoch = 1
            self.global_step = 1
            self.logger = DummyLogger()

    model = GAN(3, 28, 28)
    cb = LatentDimInterpolator(interpolate_epoch_interval=2)

    cb.on_train_epoch_end(FakeTrainer(), model)
