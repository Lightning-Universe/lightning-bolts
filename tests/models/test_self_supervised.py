import os

import pytorch_lightning as pl

from pl_bolts.datamodules import TinyCIFAR10DataModule
from pl_bolts.models.self_supervised import CPCV2, AMDIM
from pl_bolts.models.self_supervised.cpc import CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10
from tests import reset_seed


def test_cpcv2(tmpdir):
    # tmpdir = os.getcwd()
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
    # tmpdir = os.getcwd()
    reset_seed()

    model = AMDIM(data_dir=tmpdir, batch_size=2, datamodule='tiny-cifar10')
    trainer = pl.Trainer(overfit_batches=2, max_epochs=2, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0
