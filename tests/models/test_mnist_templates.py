import warnings

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.models import LitMNIST


def test_mnist(tmpdir, datadir, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".+does not have many workers which may be a bottleneck.+",
        category=PossibleUserWarning,
    )

    seed_everything(1234)

    datamodule = MNISTDataModule(data_dir=datadir, num_workers=0)
    model = LitMNIST()
    trainer = Trainer(
        limit_train_batches=0.02,
        limit_val_batches=0.02,
        max_epochs=1,
        limit_test_batches=0.02,
        default_root_dir=tmpdir,
        log_every_n_steps=5,
        accelerator="auto",
    )
    trainer.fit(model, datamodule=datamodule)
    loss = trainer.callback_metrics["train_loss"]
    assert loss <= 2.3, "mnist failed"
