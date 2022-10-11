import warnings

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised import AMDIM, BYOL, CPC_v2, Moco_v2, SimCLR, SimSiam, SwAV
from pl_bolts.models.self_supervised.cpc import CPCEvalTransformsCIFAR10, CPCTrainTransformsCIFAR10
from pl_bolts.models.self_supervised.moco.callbacks import MocoLRScheduler
from pl_bolts.models.self_supervised.moco.transforms import Moco2EvalCIFAR10Transforms, Moco2TrainCIFAR10Transforms
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.models.self_supervised.swav.transforms import SwAVEvalDataTransform, SwAVTrainDataTransform
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from tests import _MARK_REQUIRE_GPU


@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cpcv2(tmpdir, datadir):
    datamodule = CIFAR10DataModule(data_dir=datadir, num_workers=0, batch_size=2)
    datamodule.train_transforms = CPCTrainTransformsCIFAR10()
    datamodule.val_transforms = CPCEvalTransformsCIFAR10()

    model = CPC_v2(
        encoder="mobilenet_v3_small",
        patch_size=8,
        patch_overlap=2,
        online_ft=True,
        num_classes=datamodule.num_classes,
    )

    # FIXME: workaround for bug caused by
    # https://github.com/PyTorchLightning/lightning-bolts/commit/2e903c333c37ea83394c7da2ce826de1b82fb356
    model.datamodule = datamodule

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, gpus=1 if torch.cuda.device_count() > 0 else 0)
    trainer.fit(model, datamodule=datamodule)


def test_byol(tmpdir, datadir, catch_warnings):
    """Test BYOL on CIFAR-10."""
    warnings.filterwarnings(
        "ignore",
        message=".+does not have many workers which may be a bottleneck.+",
        category=PossibleUserWarning,
    )
    dm = CIFAR10DataModule(data_dir=datadir, num_workers=0, batch_size=2)
    dm.train_transforms = SimCLRTrainDataTransform(32)
    dm.val_transforms = SimCLREvalDataTransform(32)

    model = BYOL(data_dir=datadir)
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        max_epochs=1,
        accelerator="auto",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)


def test_amdim(tmpdir, datadir):
    model = AMDIM(data_dir=datadir, batch_size=2, online_ft=True, encoder="resnet18", num_workers=0)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model)


def test_moco(tmpdir, datadir):
    datamodule = CIFAR10DataModule(data_dir=datadir, num_workers=0, batch_size=2)
    datamodule.train_transforms = Moco2TrainCIFAR10Transforms()
    datamodule.val_transforms = Moco2EvalCIFAR10Transforms()

    model = Moco_v2(data_dir=datadir, batch_size=2, online_ft=True)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, callbacks=[MocoLRScheduler()])
    trainer.fit(model, datamodule=datamodule)


def test_simclr(tmpdir, datadir):
    datamodule = CIFAR10DataModule(data_dir=datadir, num_workers=0, batch_size=2)
    datamodule.train_transforms = SimCLRTrainDataTransform(32)
    datamodule.val_transforms = SimCLREvalDataTransform(32)

    model = SimCLR(batch_size=2, num_samples=datamodule.num_samples, gpus=0, nodes=1, dataset="cifar10")
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, datamodule=datamodule)


def test_swav(tmpdir, datadir, catch_warnings):
    """Test SWAV on CIFAR-10."""
    warnings.filterwarnings(
        "ignore",
        message=".+does not have many workers which may be a bottleneck.+",
        category=PossibleUserWarning,
    )
    batch_size = 2
    datamodule = CIFAR10DataModule(data_dir=datadir, batch_size=batch_size, num_workers=0)

    datamodule.train_transforms = SwAVTrainDataTransform(
        normalize=cifar10_normalization(), size_crops=[32, 16], nmb_crops=[2, 1], gaussian_blur=False
    )
    datamodule.val_transforms = SwAVEvalDataTransform(
        normalize=cifar10_normalization(), size_crops=[32, 16], nmb_crops=[2, 1], gaussian_blur=False
    )
    if torch.cuda.device_count() >= 1:
        devices = torch.cuda.device_count()
        accelerator = "gpu"
    else:
        devices = None
        accelerator = "cpu"

    model = SwAV(
        arch="resnet18",
        hidden_mlp=512,
        nodes=1,
        gpus=0 if devices is None else devices,
        num_samples=datamodule.num_samples,
        batch_size=batch_size,
        nmb_crops=[2, 1],
        sinkhorn_iterations=1,
        nmb_prototypes=2,
        queue_length=0,
        maxpool1=False,
        first_conv=False,
        dataset="cifar10",
    )
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        fast_dev_run=True,
        default_root_dir=tmpdir,
        log_every_n_steps=1,
        max_epochs=1,
    )
    trainer.fit(model, datamodule=datamodule)


def test_simsiam(tmpdir, datadir, catch_warnings):
    """Test SimSiam on CIFAR-10."""
    warnings.filterwarnings(
        "ignore",
        message=".+does not have many workers which may be a bottleneck.+",
        category=PossibleUserWarning,
    )
    dm = CIFAR10DataModule(data_dir=datadir, num_workers=0, batch_size=2)
    dm.train_transforms = SimCLRTrainDataTransform(32)
    dm.val_transforms = SimCLREvalDataTransform(32)

    model = SimSiam()
    trainer = Trainer(
        fast_dev_run=True,
        default_root_dir=tmpdir,
        max_epochs=1,
        accelerator="auto",
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)
