"""
Layer-wise adaptive rate scaling for SGD in PyTorch!
Adapted from: https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
"""
from typing import Iterable, Callable

import torch
from torch.optim.optimizer import Optimizer


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


REQUIRED = _RequiredParameter()


class LARS(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: base learning rate (\gamma_0)
        momentum: momentum factor (default: 0) ("m")
        weight_decay: weight decay (L2 penalty) (default: 0) ("\beta")
        eta: LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.

    Reference:
        Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
        Large Batch Training of Convolutional Networks: https://arxiv.org/abs/1708.03888

    Example:

        optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        optimizer.zero_grad()
        loss_fn(model(input), target).backward()
        optimizer.step()

    """

    def __init__(self,
                 params: Iterable,
                 lr: float = REQUIRED,
                 momentum: float = .9,
                 weight_decay: float = 0.0005,
                 eta: float = 0.001,
                 max_epoch: int = 200):
        if lr is not REQUIRED and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if eta < 0.0:
            raise ValueError(f"Invalid LARS coefficient value: {eta}")

        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch)
        super(LARS, self).__init__(params, defaults)

    def step(self, epoch=None, closure: Callable = None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                weight_norm = torch.norm(p.data)
                grad_norm = torch.norm(d_p)

                # Global LR computed on polynomial decay schedule
                decay = (1 - float(epoch) / max_epoch) ** 2
                global_lr = lr * decay

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)

                # Update the momentum term
                actual_lr = local_lr * global_lr

                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
                buf.mul_(momentum).add(actual_lr, d_p + weight_decay * p.data)
                p.data.add_(-buf)

        return loss
