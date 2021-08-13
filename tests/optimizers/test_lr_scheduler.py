import math

import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

EPSILON = 1e-12


class SchedulerTestNet(torch.nn.Module):
    """adapted from: https://github.com/pytorch/pytorch/blob/master/test/test_optim.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestLRScheduler:
    """adapted from: https://github.com/pytorch/pytorch/blob/master/test/test_optim.py."""

    def __init__(self, base_lr=0.05, multiplier=10):
        self.base_lr = base_lr
        self.multiplier = multiplier

        self.net = SchedulerTestNet()
        self.optimizer = SGD(
            [
                {
                    "params": self.net.conv1.parameters(),
                },
                {
                    "params": self.net.conv2.parameters(),
                    "lr": base_lr * multiplier,
                },
            ],
            lr=base_lr,
        )

        self.closed_form_net = SchedulerTestNet()
        self.closed_form_opt = SGD(
            [
                {
                    "params": self.closed_form_net.conv1.parameters(),
                },
                {
                    "params": self.closed_form_net.conv2.parameters(),
                    "lr": base_lr * multiplier,
                },
            ],
            lr=base_lr,
        )

    def _test_lr(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.optimizer.param_groups, targets):
                assert (
                    abs(target[epoch] - param_group["lr"]) < EPSILON
                ), "LR is wrong in epoch {}: expected {}, got {}".format(epoch, target[epoch], param_group["lr"])
            for scheduler in schedulers:
                scheduler.step()

    def _test_against_closed_form(self, scheduler, closed_form_scheduler, epochs=10):
        targets = []
        for epoch in range(epochs):
            closed_form_scheduler.step(epoch)
            targets.append([group["lr"] for group in self.closed_form_opt.param_groups])

        for epoch in range(epochs):
            for i, param_group in enumerate(self.optimizer.param_groups):
                assert (
                    abs(targets[epoch][i] - param_group["lr"]) < EPSILON
                ), "LR is wrong in epoch {}: expected {}, got {}".format(epoch, targets[epoch][i], param_group["lr"])
            scheduler.step()


def test_lwca_lr():
    seed_everything()

    warmup_start_lr = 0.0
    base_lr = 0.4
    eta_min = 0.0
    warmup_epochs = 6
    max_epochs = 15
    multiplier = 10

    # define target schedule
    targets = []

    # param-group1
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr, warmup_epochs)
    iters = np.arange(max_epochs - warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    # param-group2
    base_lr2 = base_lr * multiplier
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr2, warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr2 - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_lr(scheduler, targets, epochs=max_epochs)


def test_lwca_lr_with_nz_start_lr():
    seed_everything()

    warmup_start_lr = 0.2
    base_lr = 0.8
    eta_min = 0.0
    warmup_epochs = 9
    max_epochs = 28
    multiplier = 10

    # define target schedule
    targets = []

    # param-group1
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr, warmup_epochs)
    iters = np.arange(max_epochs - warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    # param-group2
    base_lr2 = base_lr * multiplier
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr2, warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr2 - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_lr(scheduler, targets, epochs=max_epochs)


def test_lwca_lr_with_nz_eta_min():
    seed_everything()

    warmup_start_lr = 0.0
    base_lr = 0.04
    eta_min = 0.0001
    warmup_epochs = 15
    max_epochs = 47
    multiplier = 17

    # define target schedule
    targets = []

    # param-group1
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr, warmup_epochs)
    iters = np.arange(max_epochs - warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    # param-group2
    base_lr2 = base_lr * multiplier
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr2, warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr2 - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_lr(scheduler, targets, epochs=max_epochs)


def test_lwca_lr_with_nz_start_lr_nz_eta_min():
    seed_everything()

    warmup_start_lr = 0.009
    base_lr = 0.07
    eta_min = 0.003
    warmup_epochs = 15
    max_epochs = 115
    multiplier = 32

    # define target schedule
    targets = []

    # param-group1
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr, warmup_epochs)
    iters = np.arange(max_epochs - warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    # param-group2
    base_lr2 = base_lr * multiplier
    warmup_lr_schedule = np.linspace(warmup_start_lr, base_lr2, warmup_epochs)
    cosine_lr_schedule = np.array(
        [
            eta_min + 0.5 * (base_lr2 - eta_min) * (1 + math.cos(math.pi * t / (max_epochs - warmup_epochs)))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    targets.append(list(lr_schedule))

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_lr(scheduler, targets, epochs=max_epochs)


def test_closed_form_lwca_lr():
    seed_everything()

    warmup_start_lr = 0.0
    base_lr = 0.4
    eta_min = 0.0
    warmup_epochs = 6
    max_epochs = 15
    multiplier = 10

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    closed_form_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.closed_form_opt,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_against_closed_form(scheduler, closed_form_scheduler, epochs=max_epochs)


def test_closed_form_lwca_lr_with_nz_start_lr():
    seed_everything()

    warmup_start_lr = 0.2
    base_lr = 0.8
    eta_min = 0.0
    warmup_epochs = 9
    max_epochs = 28
    multiplier = 10

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    closed_form_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.closed_form_opt,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_against_closed_form(scheduler, closed_form_scheduler, epochs=max_epochs)


def test_closed_form_lwca_lr_with_nz_eta_min():
    seed_everything()

    warmup_start_lr = 0.0
    base_lr = 0.04
    eta_min = 0.0001
    warmup_epochs = 15
    max_epochs = 47
    multiplier = 17

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    closed_form_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.closed_form_opt,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_against_closed_form(scheduler, closed_form_scheduler, epochs=max_epochs)


def test_closed_form_lwca_lr_with_nz_start_lr_nz_eta_min():
    seed_everything()

    warmup_start_lr = 0.009
    base_lr = 0.07
    eta_min = 0.003
    warmup_epochs = 15
    max_epochs = 115
    multiplier = 32

    test_lr_scheduler = TestLRScheduler(base_lr=base_lr, multiplier=multiplier)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    closed_form_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=test_lr_scheduler.closed_form_opt,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        warmup_start_lr=warmup_start_lr,
        eta_min=eta_min,
    )

    test_lr_scheduler._test_against_closed_form(scheduler, closed_form_scheduler, epochs=max_epochs)
