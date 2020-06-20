import pytorch_lightning as pl
from pl_bolts.models.self_supervised import CPCV2, AMDIM, SimCLR, MocoV2
from tests import reset_seed
from argparse import Namespace, ArgumentParser


def test_cpcv2(tmpdir):
    reset_seed()

    model = CPCV2(data_dir=tmpdir)
    trainer = pl.Trainer(overfit_batches=2, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0


def test_amdim(tmpdir):
    reset_seed()

    model = AMDIM(data_dir=tmpdir)
    trainer = pl.Trainer(overfit_batches=2, default_root_dir=tmpdir)
    trainer.fit(model)
    loss = trainer.callback_metrics['loss']

    assert loss > 0
