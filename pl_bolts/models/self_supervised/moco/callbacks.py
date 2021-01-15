import math

from pytorch_lightning import Callback


class MocoLRScheduler(Callback):

    def __init__(self, initial_lr=0.03, use_cosine_scheduler=False, schedule=(120, 160), max_epochs=200):
        super().__init__()
        self.lr = initial_lr
        self.use_cosine_scheduler = use_cosine_scheduler
        self.schedule = schedule
        self.max_epochs = max_epochs

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        lr = self.lr

        if self.use_cosine_scheduler:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / self.max_epochs))
        else:  # stepwise lr schedule
            for milestone in self.schedule:
                lr *= 0.1 if epoch >= milestone else 1.

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
