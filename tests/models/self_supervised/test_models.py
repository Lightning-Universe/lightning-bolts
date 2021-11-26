from distutils.version import LooseVersion

import pytest
import torch
from pytorch_lightning import Trainer

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


# todo: some pickling issue with min config
@pytest.mark.skipif(LooseVersion(torch.__version__) < LooseVersion("1.7.0"), reason="Pickling issue")
def test_byol(tmpdir, datadir):
    datamodule = CIFAR10DataModule(data_dir=datadir, num_workers=0, batch_size=2)
    datamodule.train_transforms = CPCTrainTransformsCIFAR10()
    datamodule.val_transforms = CPCEvalTransformsCIFAR10()

    model = BYOL(data_dir=datadir, num_classes=datamodule)
    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, datamodule=datamodule)


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


def test_swav(tmpdir, datadir, batch_size=2):
    # inputs, y = batch  (doesn't receive y for some reason)
    datamodule = CIFAR10DataModule(data_dir=datadir, batch_size=batch_size, num_workers=0)

    datamodule.train_transforms = SwAVTrainDataTransform(
        normalize=cifar10_normalization(), size_crops=[32, 16], nmb_crops=[2, 1], gaussian_blur=False
    )
    datamodule.val_transforms = SwAVEvalDataTransform(
        normalize=cifar10_normalization(), size_crops=[32, 16], nmb_crops=[2, 1], gaussian_blur=False
    )

    model = SwAV(
        arch="resnet18",
        hidden_mlp=512,
        gpus=0,
        nodes=1,
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

    trainer = Trainer(gpus=0, fast_dev_run=True, default_root_dir=tmpdir)

    trainer.fit(model, datamodule=datamodule)


def test_simsiam(tmpdir, datadir):
    datamodule = CIFAR10DataModule(data_dir=datadir, num_workers=0, batch_size=2)
    datamodule.train_transforms = SimCLRTrainDataTransform(32)
    datamodule.val_transforms = SimCLREvalDataTransform(32)

    model = SimSiam(batch_size=2, num_samples=datamodule.num_samples, gpus=0, nodes=1, dataset="cifar10")
    trainer = Trainer(gpus=0, fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, datamodule=datamodule)
