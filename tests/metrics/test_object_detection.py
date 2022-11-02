"""Test Object Detection Metric Functions."""

import pytest
import torch

from pl_bolts.metrics.object_detection import giou, iou


@pytest.mark.parametrize(
    "preds, target, expected_iou",
    [(torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 100, 200, 200]]), torch.tensor([[1.0]]))],
)
def test_iou_complete_overlap(preds, target, expected_iou):
    torch.testing.assert_close(iou(preds, target), expected_iou)


@pytest.mark.parametrize(
    "preds, target, expected_iou",
    [
        (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 200, 200, 300]]), torch.tensor([[0.0]])),
        (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[200, 200, 300, 300]]), torch.tensor([[0.0]])),
    ],
)
def test_iou_no_overlap(preds, target, expected_iou):
    torch.testing.assert_close(iou(preds, target), expected_iou)


@pytest.mark.parametrize(
    "preds, target, expected_iou",
    [
        (
            torch.tensor([[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]),
            torch.tensor([[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]),
            torch.tensor([[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    ],
)
def test_iou_multi(preds, target, expected_iou):
    torch.testing.assert_close(iou(preds, target), expected_iou)


@pytest.mark.parametrize(
    "preds, target, expected_giou",
    [(torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 100, 200, 200]]), torch.tensor([[1.0]]))],
)
def test_complete_overlap(preds, target, expected_giou):
    torch.testing.assert_close(giou(preds, target), expected_giou)


@pytest.mark.parametrize(
    "preds, target, expected_giou",
    [
        (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 200, 200, 300]]), torch.tensor([[0.0]])),
        (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[200, 200, 300, 300]]), torch.tensor([[-0.5]])),
    ],
)
def test_no_overlap(preds, target, expected_giou):
    torch.testing.assert_close(giou(preds, target), expected_giou)


@pytest.mark.parametrize(
    "preds, target, expected_giou",
    [
        (
            torch.tensor([[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]),
            torch.tensor([[0, 0, 100, 100], [0, 0, 50, 50], [200, 200, 300, 300]]),
            torch.tensor([[1.0, 0.25, -0.7778], [0.25, 1.0, -0.8611], [-0.7778, -0.8611, 1.0]]),
        )
    ],
)
def test_giou_multi(preds, target, expected_giou):
    torch.testing.assert_close(giou(preds, target), expected_giou, atol=0.0001, rtol=0.0001)
