import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.autoencoders import AE, resnet18_decoder, resnet18_encoder, resnet50_encoder, VAE


@pytest.mark.parametrize("dm_cls", [pytest.param(CIFAR10DataModule, id="cifar10")])
def test_vae(tmpdir, datadir, dm_cls):
    seed_everything()

    dm = dm_cls(data_dir=datadir, batch_size=4)
    model = VAE(input_height=dm.size()[-1])
    trainer = pl.Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        gpus=None,
    )

    result = trainer.fit(model, datamodule=dm)
    assert result == 1


@pytest.mark.parametrize("dm_cls", [pytest.param(CIFAR10DataModule, id="cifar10")])
def test_ae(tmpdir, datadir, dm_cls):
    seed_everything()

    dm = dm_cls(data_dir=datadir, batch_size=4)
    model = AE(input_height=dm.size()[-1])
    trainer = pl.Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        gpus=None,
    )

    result = trainer.fit(model, datamodule=dm)
    assert result == 1


@torch.no_grad()
def test_encoder():
    img = torch.rand(16, 3, 224, 224)

    encoder1 = resnet18_encoder(first_conv=False, maxpool1=True)
    encoder2 = resnet50_encoder(first_conv=False, maxpool1=True)

    out1 = encoder1(img)
    out2 = encoder2(img)

    assert out1.shape == (16, 512)
    assert out2.shape == (16, 2048)


@torch.no_grad()
def test_decoder():
    latent_dim = 128
    input_height = 288  # random but has to be a multiple of 32 for first_conv=True, maxpool1=True

    decoder1 = resnet18_decoder(latent_dim=latent_dim, input_height=input_height, first_conv=True, maxpool1=True)
    decoder2 = resnet18_decoder(latent_dim=latent_dim, input_height=input_height, first_conv=True, maxpool1=False)
    decoder3 = resnet18_decoder(latent_dim=latent_dim, input_height=input_height, first_conv=False, maxpool1=True)
    decoder4 = resnet18_decoder(latent_dim=latent_dim, input_height=input_height, first_conv=False, maxpool1=False)

    z = torch.rand(2, latent_dim)

    out1 = decoder1(z)
    out2 = decoder2(z)
    out3 = decoder3(z)
    out4 = decoder4(z)

    assert out1.shape == (2, 3, 288, 288)
    assert out2.shape == (2, 3, 288, 288)
    assert out3.shape == (2, 3, 288, 288)
    assert out4.shape == (2, 3, 288, 288)


def test_from_pretrained(datadir):
    dm = CIFAR10DataModule(data_dir=datadir, batch_size=4)
    dm.setup()
    dm.prepare_data()

    data_loader = dm.val_dataloader()

    vae = VAE(input_height=32)
    ae = AE(input_height=32)

    assert len(VAE.pretrained_weights_available()) > 0
    assert len(AE.pretrained_weights_available()) > 0

    exception_raised = False

    try:
        vae = vae.from_pretrained('cifar10-resnet18')

        # test forward method on pre-trained weights
        for x, y in data_loader:
            vae(x)
            break

        vae = vae.from_pretrained('stl10-resnet18')  # try loading weights not compatible with exact architecture

        ae = ae.from_pretrained('cifar10-resnet18')

        # test forward method on pre-trained weights
        with torch.no_grad():
            for x, y in data_loader:
                ae(x)
                break

    except Exception:
        exception_raised = True

    assert exception_raised is False, "error in loading weights"

    keyerror = False

    try:
        vae.from_pretrained('abc')
        ae.from_pretrained('xyz')
    except KeyError:
        keyerror = True

    assert keyerror is True, "KeyError not raised when provided with illegal checkpoint name"
