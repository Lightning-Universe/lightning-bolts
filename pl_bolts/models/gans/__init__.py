from pl_bolts.models.gans.basic.basic_gan_module import GAN  # noqa: F401
from pl_bolts.models.gans.dcgan.dcgan_module import DCGAN  # noqa: F401
from pl_bolts.models.gans.srgan.srgan_module import SRGAN  # noqa: F401
from pl_bolts.models.gans.srgan.srresnet_module import SRResNet  # noqa: F401

__all__ = [
    "GAN",
    "DCGAN",
    "SRGAN",
    "SRResNet",
]
