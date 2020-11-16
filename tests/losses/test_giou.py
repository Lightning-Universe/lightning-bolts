"""
Test Generalized Intersection over Union
"""

from unittest import TestCase

import torch

from pl_bolts.losses.giou import giou_loss


class TestGIoULoss(TestCase):
    def test_complete_overlap(self):
        pred = torch.tensor([[100, 100, 200, 200]])
        target = torch.tensor([[100, 100, 200, 200]])
        torch.testing.assert_allclose(giou_loss(pred, target), torch.tensor([0.0]))

    def test_no_overlap(self):
        pred = torch.tensor([[100, 100, 200, 200]])
        target = torch.tensor([[100, 200, 200, 300]])
        torch.testing.assert_allclose(giou_loss(pred, target), torch.tensor([1.0]))
        pred = torch.tensor([[100, 100, 200, 200]])
        target = torch.tensor([[200, 200, 300, 300]])
        torch.testing.assert_allclose(giou_loss(pred, target), torch.tensor([1.5]))
