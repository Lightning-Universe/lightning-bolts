import pytest
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as transform_lib

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST
from pl_bolts.models.gans import DCGAN, GAN, SRGAN, SRResNet


@pytest.mark.parametrize(
    "dm_cls",
    [
        pytest.param(MNISTDataModule, id="mnist"),
        pytest.param(CIFAR10DataModule, id="cifar10"),
    ],
)
def test_gan(tmpdir, datadir, dm_cls):
    seed_everything()

    dm = dm_cls(data_dir=datadir, num_workers=0)
    model = GAN(*dm.size())
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
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
