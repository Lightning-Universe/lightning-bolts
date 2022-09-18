import pytest
import torch
from pytorch_lightning import seed_everything

from pl_bolts.models.gans.pix2pix.components import Generator, PatchGAN


@pytest.mark.parametrize(
    "in_shape, out_shape",
    [
        pytest.param((3, 128, 128), (3, 128, 128), id="multichannel"),
        pytest.param((1, 128, 128), (3, 128, 128), id="singlechannel"),
    ],
)
def test_generator(catch_warnings, in_shape, out_shape):
    batch_dim = 10
    in_channels = in_shape.size(0)
    out_channels = out_shape.size(0)
    seed_everything(1234)
    generator = Generator(in_channels=in_channels, out_channels=out_channels)
    conditional_image = torch.randn(batch_dim, *in_shape)
    samples = generator(conditional_image)
    assert samples.shape == (batch_dim, *out_shape)


@pytest.mark.parametrize(
    "img_shape",
    [
        pytest.param((3, 128, 128), id="discriminator-multichannel"),
        pytest.param((1, 128, 128), id="discriminator-singlechannel"),
    ],
)
def test_discriminator(catch_warnings, img_shape):
    batch_dim = 10
    in_channels = img_shape.size(0)
    seed_everything(1234)
    discriminator = PatchGAN(input_channels=in_channels)
    samples = torch.randn(batch_dim, *img_shape)
    real_or_fake = discriminator(samples)
    assert real_or_fake.shape == (batch_dim, 1)
    assert (torch.clamp(real_or_fake.clone(), 0, 1) == real_or_fake).all()
