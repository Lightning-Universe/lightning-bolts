import pytorch_lightning as pl
from pl_bolts.models.self_supervised import CPCV2, AMDIM, SimCLR, MocoV2
from tests import reset_seed
from argparse import Namespace, ArgumentParser


def test_cpcv2(tmpdir):
    reset_seed()

    parser = ArgumentParser()
    parser = CPCV2.add_model_specific_args(parser)
    hparams = parser.parse_args(args=[])
    hparams.batch_size = 2
    hparams.data_dir = tmpdir
    hparams.meta_root = tmpdir

    model = CPCV2(hparams)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model)
    trainer.test()
    loss = trainer.callback_metrics['loss']

    assert loss > 0
