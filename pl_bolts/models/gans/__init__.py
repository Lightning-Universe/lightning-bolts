from pl_bolts.models.gans.basic.basic_gan_module import GAN  # noqa: F401
from pl_bolts.models.gans.dcgan.dcgan_module import DCGAN  # noqa: F401
from pl_bolts.models.gans.pix2pix.pix2pix_module import Pix2Pix

__all__ = [
    "GAN",
    "DCGAN",
    "Pix2Pix"
]
