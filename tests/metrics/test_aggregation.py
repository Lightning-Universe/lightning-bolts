"""Test Aggregation Metric Functions."""

import pytest
import torch

from pl_bolts.metrics.aggregation import accuracy, mean, precision_at_k


@pytest.mark.parametrize(
    "preds, expected_mean",
    [(torch.tensor([[100.0, 100.0, 200.0, 200.0]]), torch.tensor(150.0))],
)
def test_mean(preds, expected_mean):
    x = {"test": preds}
    torch.testing.assert_close(mean([x], "test"), expected_mean)


@pytest.mark.parametrize(
    "preds, target, expected_accuracy",
    [
        (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[2]]), torch.tensor(1.0)),
        (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[0]]), torch.tensor(0.0)),
    ],
)
def test_accuracy(preds, target, expected_accuracy):
    torch.testing.assert_close(accuracy(preds, target), expected_accuracy)


@pytest.mark.parametrize(
    "output, target, expected_precision_at_k",
    [
        (torch.tensor([[100.0, 100.0, 200.0, 200.0]]), torch.tensor([[2]]), [torch.tensor([100.0])]),
        (torch.tensor([[100.0, 100.0, 200.0, 200.0]]), torch.tensor([[1]]), [torch.tensor([0.0])]),
    ],
)
def test_precision_at_k(output, target, expected_precision_at_k):
    torch.testing.assert_close(precision_at_k(output, target), expected_precision_at_k)
