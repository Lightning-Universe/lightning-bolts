import pytorch_lightning as pl
from tests import reset_seed

from pl_bolts.models.self_supervised import MocoV2
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.moco.transforms import Moco2TrainCIFAR10Transforms, Moco2EvalCIFAR10Transforms
from pl_bolts.models.self_supervised.moco.callbacks import MocoLRScheduler


def test_moco(tmpdir):
    reset_seed()

    datamodule = CIFAR10DataModule(tmpdir, num_workers=0)
    datamodule.train_transforms = Moco2TrainCIFAR10Transforms()
    datamodule.val_transforms = Moco2EvalCIFAR10Transforms()

    model = MocoV2(data_dir=tmpdir, batch_size=2, datamodule=datamodule, online_ft=True)
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir, callbacks=[MocoLRScheduler()])
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0
