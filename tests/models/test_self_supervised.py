import os

import pytorch_lightning as pl

from pl_bolts.models.self_supervised import CPCV2, AMDIM, MocoV2, SimCLR
from pl_bolts.datamodules import TinyCIFAR10DataModule
from pl_bolts.models.self_supervised.cpc import CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10
from tests import reset_seed
from pl_bolts.models.self_supervised.moco.transforms import (Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms)
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform


def test_cpcv2(tmpdir):
    reset_seed()

    datamodule = TinyCIFAR10DataModule(data_dir=tmpdir)
    datamodule.train_transforms = CPCTrainTransformsCIFAR10()
    datamodule.val_transforms = CPCEvalTransformsCIFAR10()

    model = CPCV2(encoder='resnet18', data_dir=tmpdir, batch_size=2, datamodule=datamodule)
    trainer = pl.Trainer(overfit_batches=2, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0


def test_amdim(tmpdir):
    reset_seed()

    model = AMDIM(data_dir=tmpdir, batch_size=2)
    trainer = pl.Trainer(overfit_batches=2, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0


def test_moco(tmpdir):
    reset_seed()

    datamodule = TinyCIFAR10DataModule(tmpdir)
    datamodule.train_transforms = Moco2TrainCIFAR10Transforms()
    datamodule.val_transforms = Moco2EvalCIFAR10Transforms()

    model = MocoV2(data_dir=tmpdir, batch_size=2, datamodule=datamodule)
    trainer = pl.Trainer(overfit_batches=2, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0


def test_simclr(tmpdir):
    reset_seed()

    datamodule = TinyCIFAR10DataModule(tmpdir)
    datamodule.train_transforms = SimCLRTrainDataTransform(32)
    datamodule.val_transforms = SimCLREvalDataTransform(32)

    model = SimCLR(data_dir=tmpdir, batch_size=2, datamodule=datamodule)
    trainer = pl.Trainer(overfit_batches=2, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0