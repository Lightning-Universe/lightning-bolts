import pytorch_lightning as pl
from tests import reset_seed

from pl_bolts.models.self_supervised import CPCV2
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.cpc import CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10


def test_cpcv2(tmpdir):
    reset_seed()

    datamodule = CIFAR10DataModule(data_dir=tmpdir, num_workers=0)
    datamodule.train_transforms = CPCTrainTransformsCIFAR10()
    datamodule.val_transforms = CPCEvalTransformsCIFAR10()

    model = CPCV2(encoder='resnet18', data_dir=tmpdir, batch_size=2, online_ft=True, datamodule=datamodule)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0
