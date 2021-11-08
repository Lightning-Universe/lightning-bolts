from pytorch_lightning import Trainer, seed_everything

from pl_bolts.models import LitMNIST


def test_mnist(tmpdir, datadir):
    seed_everything()

    model = LitMNIST(data_dir=datadir, num_workers=0)
    trainer = Trainer(
        limit_train_batches=0.01,
        limit_val_batches=0.01,
        max_epochs=1,
        limit_test_batches=0.01,
        default_root_dir=tmpdir,
    )
    trainer.fit(model)
    loss = trainer.callback_metrics["train_loss"]
    assert loss <= 2.2, "mnist failed"
