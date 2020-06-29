from pl_bolts.datamodules.cifar10_dataset import CIFAR10
from pl_bolts.datamodules import DACDataModule


def test_dev_datasets(tmpdir):

    ds = CIFAR10(tmpdir)
    for b in ds:
        pass


def test_dac_dm(tmpdir):

    dm = DACDataModule(data_dir=tmpdir, use_tiny_dac=True)
    dm.prepare_data()

    for b in dm.train_dataloader():
        break
    for b in dm.val_dataloader():
        break
    for b in dm.test_dataloader():
        break
