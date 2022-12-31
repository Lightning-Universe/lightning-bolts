import warnings

import pytest
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as transform_lib

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST
from pl_bolts.models.gans import DCGAN, GAN, SRGAN, SRResNet


@pytest.mark.parametrize(
    "dm_cls",
    [pytest.param(MNISTDataModule, id="mnist"), pytest.param(CIFAR10DataModule, id="cifar10")],
)
def test_gan(tmpdir, datadir, catch_warnings, dm_cls):
    # Validation loop for GANs is not well defined!
    warnings.filterwarnings(
        "ignore",
        message="You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The dataloader, train_dataloader, does not have many workers which may be a bottleneck",
        category=PossibleUserWarning,
    )
    seed_everything(1234)
    dm = dm_cls(data_dir=datadir, num_workers=0)
    model = GAN(*dm.dims)

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=-1,
        fast_dev_run=True,
        log_every_n_steps=1,
        accelerator="auto",
        # TODO: We need to be able to support multiple GPUs in such a simple scenario.
        # But, DDP is throwing ugly errors at me at the moment
        devices=1,
    )
    trainer.fit(model, datamodule=dm)


@pytest.mark.parametrize(
    "dm_cls", [pytest.param(MNISTDataModule, id="mnist"), pytest.param(CIFAR10DataModule, id="cifar10")]
)
def test_dcgan(tmpdir, datadir, dm_cls):
    seed_everything()

    transforms = transform_lib.Compose([transform_lib.Resize(64), transform_lib.ToTensor()])
    dm = dm_cls(data_dir=datadir, train_transforms=transforms, val_transforms=transforms, test_transforms=transforms)

    model = DCGAN(image_channels=dm.dims[0])
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)


@pytest.mark.parametrize("sr_module_cls", [SRResNet, SRGAN])
@pytest.mark.parametrize("scale_factor", [2, 4])
def test_sr_modules(tmpdir, datadir, sr_module_cls, scale_factor):
    seed_everything(42)

    dl = DataLoader(SRMNIST(scale_factor, root=datadir, download=True))
    model = sr_module_cls(image_channels=1, scale_factor=scale_factor)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dl)
