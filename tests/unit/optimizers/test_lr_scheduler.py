import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

from tests import reset_seed
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class SchedulerTestNet(torch.nn.Module):
    """
    adapted from: https://github.com/pytorch/pytorch/blob/master/test/test_optim.py
    """
    def __init__(self):
        super(SchedulerTestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestLRScheduler(object):
    """
    adapted from: https://github.com/pytorch/pytorch/blob/master/test/test_optim.py
    """
    def __init__(self):
        self.net = SchedulerTestNet()
        self.opt = SGD([{'params': self.net.conv1.parameters()},
                        {'params': self.net.conv2.parameters(), 'lr': 0.5}], lr=0.05)

    def _test_lr(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                # TODO: re-write assert-equal
                self.assertEqual(target[epoch], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, target[epoch], param_group['lr']), atol=1e-5, rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def _test_against_closed_form(self, scheduler, closed_form_scheduler, epochs=10):
        self.setUp()
        targets = []
        for epoch in range(epochs):
            closed_form_scheduler.step(epoch)
            targets.append([group['lr'] for group in self.opt.param_groups])
        self.setUp()
        for epoch in range(epochs):
            scheduler.step()
            for i, param_group in enumerate(self.opt.param_groups):
                self.assertEqual(targets[epoch][i], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, targets[epoch][i], param_group['lr']), atol=1e-5, rtol=0)


def test_lwca_lr(tmpdir):
    reset_seed()

"""
# Test for non-zero start lr, non-zero end lr
# test closed form

1) 0 start lr, 0 end lr, random base_lr
2) non-zero start lr, 0 end lr, random base_lr
3) 0 start lr, non-zero end lr, random base_lr
4) non-zero start lr, non-zero end lr, random base_lr

5) repeat for closed form
"""
