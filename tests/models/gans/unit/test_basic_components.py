import pytest
import torch
from pytorch_lightning import seed_everything

from pl_bolts.models.gans.basic.components import Discriminator, Generator


@pytest.mark.parametrize(
    "latent_dim, img_shape",
    [
        pytest.param(100, (3, 28, 28), id="100-multichannel"),
        pytest.param(100, (1, 28, 28), id="100-singlechannel"),
    ],
)
def test_generator(catch_warnings, latent_dim, img_shape):
    batch_dim = 10
    seed_everything(1234)
    generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
    noise = torch.randn(batch_dim, latent_dim)
    samples = generator(noise)
    assert samples.shape == (batch_dim, *img_shape)


@pytest.mark.parametrize(
    "img_shape",
    [
        pytest.param((3, 28, 28), id="discriminator-multichannel"),
        pytest.param((1, 28, 28), id="discriminator-singlechannel"),
    ],
)
def test_discriminator(catch_warnings, img_shape):
    batch_dim = 10
    seed_everything(1234)
    discriminator = Discriminator(img_shape=img_shape)
    samples = torch.randn(batch_dim, *img_shape)
    real_or_fake = discriminator(samples)
    assert real_or_fake.shape == (batch_dim, 1)
    assert (torch.clamp(real_or_fake.clone(), 0, 1) == real_or_fake).all()
