import pytest
import torch
from pl_bolts.models.gans.pix2pix.components import Generator, PatchGAN
from pytorch_lightning import seed_everything


@pytest.mark.parametrize(
    ("in_shape", "out_shape"),
    [
        pytest.param((3, 256, 256), (3, 256, 256), id="multichannel"),
        pytest.param((1, 256, 256), (3, 256, 256), id="singlechannel"),
        pytest.param((3, 128, 128), (3, 128, 128), id="img shape (3, 128, 128), (3, 128, 128)"),
        pytest.param((1, 128, 128), (3, 128, 128), id="img shape (1, 128, 128), (3, 128, 128)"),
        pytest.param((3, 64, 64), (3, 64, 64), id="img shape (3, 64, 64), (3, 64, 64)"),
        pytest.param((1, 64, 64), (3, 64, 64), id="img shape (1, 64, 64), (3, 64, 64)"),
    ],
)
def test_generator(catch_warnings, in_shape, out_shape):
    batch_dim = 10
    in_channels = in_shape[0]
    out_channels = out_shape[0]
    seed_everything(1234)
    generator = Generator(in_channels=in_channels, out_channels=out_channels)
    condition_image = torch.randn(batch_dim, *in_shape)

    samples = generator(condition_image)
    assert samples.shape == (batch_dim, *out_shape)


@pytest.mark.parametrize(
    ("in_shape", "out_shape"),
    [
        pytest.param((3, 256, 256), (3, 256, 256), id="discriminator-multichannel"),
        pytest.param((1, 256, 256), (3, 256, 256), id="discriminator-singlechannel"),
        pytest.param((3, 128, 128), (3, 128, 128), id="img shape (3, 128, 128), (3, 128, 128)"),
        pytest.param((1, 128, 128), (3, 128, 128), id="img shape (1, 128, 128), (3, 128, 128)"),
        pytest.param((3, 64, 64), (3, 64, 64), id="img shape (3, 64, 64), (3, 64, 64)"),
        pytest.param((1, 64, 64), (3, 64, 64), id="img shape (1, 64, 64), (3, 64, 64)"),
    ],
)
def test_discriminator(catch_warnings, in_shape, out_shape):
    batch_dim = 10
    in_channels = in_shape[0]
    out_channels = out_shape[0]
    seed_everything(1234)
    discriminator = PatchGAN(input_channels=in_channels + out_channels)
    condition_image = torch.randn(batch_dim, *in_shape)
    real_image = torch.randn(batch_dim, *out_shape)

    real_or_fake = discriminator(condition_image, real_image)
    assert real_or_fake.shape == (batch_dim, 1, in_shape[1] // 16, in_shape[2] // 16)
    assert (torch.clamp(real_or_fake.clone(), 0, 1) == real_or_fake).all()
