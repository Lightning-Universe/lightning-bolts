"""
Test Object Detection Metric Functions
"""

import pytest
import torch

from pl_bolts.metrics.object_detection import giou, iou


@pytest.mark.parametrize("preds, target, expected_iou", [
    (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 100, 200, 200]]), torch.tensor([1.0]))
])
def test_iou_complete_overlap(preds, target, expected_iou):
    torch.testing.assert_allclose(iou(preds, target), expected_iou)


@pytest.mark.parametrize("preds, target, expected_iou", [
    (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 200, 200, 300]]), torch.tensor([0.0])),
    (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[200, 200, 300, 300]]), torch.tensor([0.0])),
])
def test_iou_no_overlap(preds, target, expected_iou):
    torch.testing.assert_allclose(iou(preds, target), expected_iou)


@pytest.mark.parametrize("preds, target, expected_giou", [
    (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 100, 200, 200]]), torch.tensor([1.0]))
])
def test_complete_overlap(preds, target, expected_giou):
    torch.testing.assert_allclose(giou(preds, target), expected_giou)


@pytest.mark.parametrize("preds, target, expected_giou", [
    (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[100, 200, 200, 300]]), torch.tensor([0.0])),
    (torch.tensor([[100, 100, 200, 200]]), torch.tensor([[200, 200, 300, 300]]), torch.tensor([-0.5])),
])
def test_no_overlap(preds, target, expected_giou):
    torch.testing.assert_allclose(giou(preds, target), expected_giou)
