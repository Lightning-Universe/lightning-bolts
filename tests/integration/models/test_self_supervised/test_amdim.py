import pytorch_lightning as pl
from tests import reset_seed

from pl_bolts.models.self_supervised import AMDIM


def test_amdim(tmpdir):
    reset_seed()

    model = AMDIM(data_dir=tmpdir, batch_size=2, online_ft=True, encoder='resnet18')
    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0
