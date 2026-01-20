from copy import deepcopy

import pytest
import torch
from torch import nn

from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate


@pytest.mark.parametrize("initial_tau", [-0.1, 0.0, 0.996, 1.0, 1.1])
def test_byol_ma_weight_single_update_callback(initial_tau, catch_warnings):
    """Check BYOL exponential moving average weight update rule for a single update."""
    if 0.0 <= initial_tau <= 1.0:
        # Create simple one layer network and their copies
        online_network = nn.Linear(100, 10)
        target_network = deepcopy(online_network)
        online_network_copy = deepcopy(online_network)
        target_network_copy = deepcopy(target_network)

        # Check parameters are equal
        assert torch.equal(next(iter(online_network.parameters()))[0], next(iter(target_network.parameters()))[0])

        # Simulate weight update
        opt = torch.optim.SGD(online_network.parameters(), lr=0.1)
        y = online_network(torch.randn(3, 100))
        loss = y.sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Check online network update
        assert not torch.equal(
            next(iter(online_network.parameters()))[0], next(iter(online_network_copy.parameters()))[0]
        )

        # Update target network weights via callback
        cb = BYOLMAWeightUpdate(initial_tau)
        cb.update_weights(online_network, target_network)

        # Check target network update according to value of tau
        if initial_tau == 0.0:
            assert torch.equal(next(iter(target_network.parameters()))[0], next(iter(online_network.parameters()))[0])
        elif initial_tau == 1.0:
            assert torch.equal(
                next(iter(target_network.parameters()))[0], next(iter(target_network_copy.parameters()))[0]
            )
        else:
            for online_p, target_p in zip(online_network.parameters(), target_network_copy.parameters()):
                target_p.data = initial_tau * target_p.data + (1.0 - initial_tau) * online_p.data

            assert torch.equal(
                next(iter(target_network.parameters()))[0], next(iter(target_network_copy.parameters()))[0]
            )
    else:
        with pytest.raises(ValueError, match="initial tau should be"):
            cb = BYOLMAWeightUpdate(initial_tau)
