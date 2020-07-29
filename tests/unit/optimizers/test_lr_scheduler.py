import pytorch_lightning as pl
from tests import reset_seed

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


def test_lin_warmup_cos_anneal_lr(tmpdir):
    reset_seed()
