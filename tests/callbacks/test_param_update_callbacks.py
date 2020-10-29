from copy import deepcopy

import torch
from torch import nn

from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate


def test_byol_ma_weight_update_callback():
    a = nn.Linear(100, 10)
    b = deepcopy(a)
    a_original = deepcopy(a)
    b_original = deepcopy(b)

    # make sure a params and b params are the same
    assert torch.equal(next(iter(a.parameters()))[0], next(iter(b.parameters()))[0])

    # fake weight update
    opt = torch.optim.SGD(a.parameters(), lr=0.1)
    y = a(torch.randn(3, 100))
    loss = y.sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    # make sure a did in fact update
    assert not torch.equal(next(iter(a_original.parameters()))[0], next(iter(a.parameters()))[0])

    # do update via callback
    cb = BYOLMAWeightUpdate(0.8)
    cb.update_weights(a, b)

    assert not torch.equal(next(iter(b_original.parameters()))[0], next(iter(b.parameters()))[0])
