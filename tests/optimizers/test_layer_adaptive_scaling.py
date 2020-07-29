import pytest

from pl_bolts.models import LitMNIST
from pl_bolts.optimizers.layer_adaptive_scaling import LARS, REQUIRED
from tests import reset_seed


def test_lars_lr_greater_than_zero(tmpdir):
    reset_seed()

    model = LitMNIST()
    with pytest.raises(ValueError, match='Invalid learning rate.*'):
        opt = LARS(model.parameters(), lr=-0.5)

    opt = LARS(model.parameters(), lr=0.003)


def test_lars_momentum_greater_than_zero(tmpdir):
    reset_seed()

    model = LitMNIST()
    with pytest.raises(ValueError, match='Invalid momentum.*'):
        opt = LARS(model.parameters(), lr=0.003, momentum=-0.01)

    opt = LARS(model.parameters(), lr=0.003, momentum=0.01)


def test_lars_weight_decay_greater_than_zero(tmpdir):
    reset_seed()

    model = LitMNIST()
    with pytest.raises(ValueError, match='Invalid weight_decay.*'):
        opt = LARS(model.parameters(), lr=0.003, weight_decay=-0.01)

    opt = LARS(model.parameters(), lr=0.003, weight_decay=0.01)


def test_lars_eta_greater_than_zero(tmpdir):
    reset_seed()

    model = LitMNIST()
    with pytest.raises(ValueError, match='Invalid LARS coefficient.*'):
        opt = LARS(model.parameters(), lr=0.003, eta=-0.01)

    opt = LARS(model.parameters(), lr=0.003, eta=0.01)
