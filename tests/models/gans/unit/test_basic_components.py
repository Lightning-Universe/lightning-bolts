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
def test_generator(latent_dim, img_shape):
    batch_dim = 10
    seed_everything()
    generator = Generator(latent_dim=latent_dim, img_shape=img_shape)
    noise = torch.randn(batch_dim, latent_dim)
    samples = generator(noise)
    assert samples.shape == (batch_dim, *img_shape)
