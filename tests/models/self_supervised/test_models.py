import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised import CPCV2, AMDIM, MocoV2, SimCLR, BYOL, SwAV
from pl_bolts.models.self_supervised.cpc import CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10
from pl_bolts.models.self_supervised.moco.callbacks import MocoLRScheduler
from pl_bolts.models.self_supervised.moco.transforms import (Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms)
from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.models.self_supervised.swav.transforms import SwAVTrainDataTransform, SwAVEvalDataTransform
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


# TODO: this test is hanging (runs for more then 10min) so we need to use GPU or optimize it...
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_cpcv2(tmpdir):
    seed_everything()

    datamodule = CIFAR10DataModule(data_dir=tmpdir, num_workers=0, batch_size=2)
    datamodule.train_transforms = CPCTrainTransformsCIFAR10()
    datamodule.val_transforms = CPCEvalTransformsCIFAR10()

    model = CPCV2(encoder='resnet18', data_dir=tmpdir, batch_size=2, online_ft=True, datamodule=datamodule)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.progress_bar_dict['val_nce']

    assert float(loss) > 0


# TODO: this test is hanging (runs for more then 10min) so we need to use GPU or optimize it...
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_byol(tmpdir):
    seed_everything()

    datamodule = CIFAR10DataModule(data_dir=tmpdir, num_workers=0, batch_size=2)
    datamodule.train_transforms = CPCTrainTransformsCIFAR10()
    datamodule.val_transforms = CPCEvalTransformsCIFAR10()

    model = BYOL(data_dir=tmpdir, num_classes=datamodule)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir, max_steps=2)
    trainer.fit(model, datamodule)
    loss = trainer.progress_bar_dict['loss']

    assert float(loss) < 1.0


def test_amdim(tmpdir):
    seed_everything()

    model = AMDIM(data_dir=tmpdir, batch_size=2, online_ft=True, encoder='resnet18')
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.progress_bar_dict['loss']

    assert float(loss) > 0


def test_moco(tmpdir):
    seed_everything()

    datamodule = CIFAR10DataModule(tmpdir, num_workers=0, batch_size=2)
    datamodule.train_transforms = Moco2TrainCIFAR10Transforms()
    datamodule.val_transforms = Moco2EvalCIFAR10Transforms()

    model = MocoV2(data_dir=tmpdir, batch_size=2, online_ft=True)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir, callbacks=[MocoLRScheduler()])
    trainer.fit(model, datamodule=datamodule)
    loss = trainer.progress_bar_dict['loss']

    assert float(loss) > 0


def test_simclr(tmpdir):
    seed_everything()

    datamodule = CIFAR10DataModule(tmpdir, num_workers=0, batch_size=2)
    datamodule.train_transforms = SimCLRTrainDataTransform(32)
    datamodule.val_transforms = SimCLREvalDataTransform(32)

    model = SimCLR(batch_size=2, num_samples=datamodule.num_samples)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model, datamodule)
    loss = trainer.progress_bar_dict['loss']

    assert float(loss) > 0


def test_swav(tmpdir):
    seed_everything(42)

    batch_size = 2

    # inputs, y = batch  (doesn't receive y for some reason)
    datamodule = CIFAR10DataModule(
        data_dir=tmpdir,
        batch_size=batch_size,
        num_workers=0
    )

    datamodule.train_transforms = SwAVTrainDataTransform(
        normalize=cifar10_normalization(),
        size_crops=[32, 16],
        nmb_crops=[2, 1],
        gaussian_blur=False
    )
    datamodule.val_transforms = SwAVEvalDataTransform(
        normalize=cifar10_normalization(),
        size_crops=[32, 16],
        nmb_crops=[2, 1],
        gaussian_blur=False
    )

    model = SwAV(
        arch='resnet18',
        hidden_mlp=512,
        gpus=0,
        num_samples=datamodule.num_samples,
        batch_size=batch_size,
        nmb_crops=[2, 1],
        sinkhorn_iterations=1,
        nmb_prototypes=2,
        maxpool1=False,
        first_conv=False,
        dataset='cifar10'
    )

    trainer = pl.Trainer(
        gpus=0, fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir, max_steps=3
    )
    trainer.fit(model, datamodule)
    loss = trainer.progress_bar_dict['loss']

    assert float(loss) > 0
