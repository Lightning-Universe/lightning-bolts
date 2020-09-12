import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.models.autoencoders import AE, VAE
from pl_bolts.models.autoencoders.basic_ae import AEEncoder
from pl_bolts.models.autoencoders.basic_vae import resnet18_encoder, resnet18_decoder


@pytest.mark.parametrize(
    "dm_cls", [pytest.param(MNISTDataModule, id="mnist"), pytest.param(CIFAR10DataModule, id="cifar10")]
)
def test_vae(tmpdir, dm_cls):
    seed_everything()
    dm = dm_cls(batch_size=2, num_workers=0)
    model = VAE(*dm.size())
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir, deterministic=True)
    trainer.fit(model, dm)
    results = trainer.test(model, datamodule=dm)[0]
    loss = results["test_loss"]

    assert loss > 0, "VAE failed"


@pytest.mark.parametrize(
    "dm_cls", [pytest.param(MNISTDataModule, id="mnist"), pytest.param(CIFAR10DataModule, id="cifar10")]
)
def test_ae(tmpdir, dm_cls):
    seed_everything()
    dm = dm_cls(batch_size=2, num_workers=0)
    model = VAE(*dm.size())
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)


@pytest.mark.parametrize(
    "hidden_dim,latent_dim,batch_size,channels,height,width",
    [
        pytest.param(128, 2, 16, 1, 28, 28, id="like-mnist-hidden-128-latent-2"),
        pytest.param(128, 4, 16, 1, 28, 28, id="like-mnist-hidden-128-latent-4"),
        pytest.param(64, 4, 16, 1, 28, 28, id="like-mnist-hidden-64-latent-4"),
        pytest.param(128, 2, 16, 3, 32, 32, id="like-cifar10-hidden-128-latent-2"),
    ],
)
def test_basic_ae_encoder(tmpdir, hidden_dim, latent_dim, batch_size, channels, height, width):
    seed_everything()
    encoder = AEEncoder(hidden_dim, latent_dim, channels, width, height)
    x = torch.randn(batch_size, channels, width, height)
    z = encoder(x)
    assert z.shape == (batch_size, latent_dim)


@pytest.mark.parametrize(
    "hidden_dim,latent_dim,batch_size,channels,height,width",
    [
        pytest.param(128, 2, 16, 1, 28, 28, id="like-mnist-hidden-128-latent-2"),
        pytest.param(128, 4, 16, 1, 28, 28, id="like-mnist-hidden-128-latent-4"),
        pytest.param(64, 4, 16, 1, 28, 28, id="like-mnist-hidden-64-latent-4"),
        pytest.param(128, 2, 16, 3, 32, 32, id="like-cifar10-hidden-128-latent-2"),
    ],
)
def test_basic_vae_components(tmpdir, hidden_dim, latent_dim, batch_size, channels, height, width):
    seed_everything()
    enc = resnet18_encoder()
    dec = resnet18_decoder()
    enc = Encoder(hidden_dim, latent_dim, channels, width, height)
    x = torch.randn(batch_size, channels, width, height)
    mu, sigma = enc(x)

    assert mu.shape == sigma.shape

    dec = Decoder(hidden_dim, latent_dim, width, height, channels)
    decoded_x = dec(mu)

    assert decoded_x.view(-1).shape == x.view(-1).shape
