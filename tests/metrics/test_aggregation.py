"""Test Object Detection Metric Functions."""

import pytest
import torch

from pl_bolts.metrics.aggregation import mean, accuracy, precision_at_k

@pytest.mark.parametrize(
    "preds, expected_mean",
    [(torch.tensor([[100., 100., 200., 200.]]), 150.0)],
)
def test_mean(preds, expected_mean):
    x = {"test":preds}
    torch.testing.assert_allclose(mean([x], "test"), expected_mean)
