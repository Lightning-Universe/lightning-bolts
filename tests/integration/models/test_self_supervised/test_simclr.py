import pytorch_lightning as pl
from tests import reset_seed

from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.simclr.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform


def test_simclr(tmpdir):
    reset_seed()

    datamodule = CIFAR10DataModule(tmpdir, num_workers=0)
    datamodule.train_transforms = SimCLRTrainDataTransform(32)
    datamodule.val_transforms = SimCLREvalDataTransform(32)

    model = SimCLR(data_dir=tmpdir, batch_size=2, datamodule=datamodule, online_ft=True)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0
