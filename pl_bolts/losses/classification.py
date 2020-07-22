import torch
from torch import nn

from pytorch_lightning.metrics.functional.reduction import reduce
from pytorch_lightning.metrics.functional import to_onehot


def _binary_focal_loss(pred, target, alpha, gamma, weight, reduction, from_logits):
    alpha = torch.ones_like(pred) * alpha
    alpha = torch.where(target.eq(1), alpha, 1 - alpha)

    if from_logits:
        pred = torch.sigmoid(pred)

    loss = F.binary_cross_entropy(pred, target, weight=weight, reduction='none')
    pred = torch.where(target.eq(1), pred, 1 - pred)

    focal_weight = (1 - pred) ** gamma
    focal_loss = alpha * focal_weight * loss
    return reduce(focal_loss, reduction)


def binary_focal_loss(pred, target, alpha=0.25, gamma=2., weight=None, reduction='elementwise_mean'):
    return _binary_focal_loss(pred, target, alpha, gamma, weight, reduction, from_logits=False)


def binary_focal_loss_with_logits(pred, target, alpha=0.25, gamma=2., weight=None, reduction='elementwise_mean'):
    return _binary_focal_loss(pred, target, alpha, gamma, weight, reduction, from_logits=True)


def _focal_loss(pred, target, alpha, gamma, weight, ignore_index, reduction, from_logits):
    if from_logits is True:
        pred = F.softmax(pred, dim=1)

    alpha = torch.ones_like(target) * alpha
    alpha = torch.where(torch.eq(target, 1), alpha, 1 - alpha).unsqueeze(1)

    loss = F.nll_loss(pred.log(), target, weight=weight, ignore_index=ignore_index, reduction='none')
    target = to_onehot(target, num_classes=pred.size(1))

    focal_weight = (1 - pred) ** gamma
    focal_loss = (alpha * focal_weight * loss.unsqueeze(1)).sum(dim=1)
    return reduce(focal_loss, reduction)


def focal_loss(pred, target, alpha=0.25, gamma=2., weight=None, ignore_index=-100, reduction='elementwise_mean'):
    return _focal_loss(pred, target, alpha, gamma, weight, ignore_index, reduction, from_logits=False)


def focal_loss_with_logits(pred, target, alpha=0.25, gamma=2., weight=None, ignore_index=-100, reduction='elementwise_mean'):
    return _focal_loss(pred, target, alpha, gamma, weight, ignore_index, reduction, from_logits=True)


def _label_smoothing_loss(pred, target, smoothing, weight, ignore_index, reduction, from_logits):
    if from_logits is True:
        pred = F.log_softmax(pred, dim=1)

    if weight is not None:
        pred = pred * weight.unsqueeze(0).to(pred)

    num_classes = pred.size(1)
    one_hot = to_onehot(target, num_classes).type(pred.dtype)
    one_hot.fill_(smoothing / (num_classes - 1))
    one_hot.scatter_(1, target.data.unsqueeze(1), 1 - smoothing)

    loss = (-one_hot * pred).sum(dim=1)
    masked_indices = target.eq(ignore_index) if ignore_index >= 0 else None

    if masked_indices is not None:
        if reduction == 'none':
            loss.masked_fill_(masked_indices, 0)
        else:
            loss = loss[~masked_indices]

    return reduce(loss, reduction)


def label_smoothing_loss(pred, target, smoothing=0.1, weight=None, ignore_index=-100, reduction='elementwise_mean'):
    return _label_smoothing_loss(pred, target, smoothing, weight, ignore_index, reduction, from_logits=False)


def label_smoothing_loss_with_logits(pred, target, smoothing=0.1, weight=None, ignore_index=-100, reduction='elementwise_mean'):
    return _label_smoothing_loss(pred, target, smoothing, weight, ignore_index, reduction, from_logits=True)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., weight=None, reduction='elementwise_mean'):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        return binary_focal_loss(pred, target, self.alpha, self.gamma, self.weight, self.reduction)


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., weight=None, reduction='elementwise_mean'):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, pred, target):
        return binary_focal_loss_with_logits(pred, target, self.alpha, self.gamma, self.weight, self.reduction)


class FocalLoss(nn.Module):
    def __init__(
            self,
            alpha=0.25,
            gamma=2.,
            weight=None,
            ignore_index=-100,
            reduction='elementwise_mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        return focal_loss(pred, target, self.alpha, self.gamma, self.weight, self.ignore_index, self.reduction)


class FocalLossWithLogits(nn.Module):
    def __init__(
            self,
            alpha=0.25,
            gamma=2.,
            weight=None,
            ignore_index=-100,
            reduction='elementwise_mean'
    ):
        super().__init__()
      
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        return focal_loss_with_logits(pred, target, self.alpha, self.gamma, self.weight, self.ignore_index, self.reduction)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, weight=None, ignore_index=-100, reduction='elementwise_mean'):
        super().__init__()

        self.smoothing = smoothing
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    def forward(self, pred, target):

        return label_smoothing_loss(pred, target, self.smoothing, self.weight, self.ignore_index, self.reduction)

class LabelSmoothingLossWithLogits(nn.Module):
    def __init__(self, smoothing=0.1, weight=None, ignore_index=-100, reduction='elementwise_mean'):
        super().__init__()

        self.smoothing = smoothing
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred, target):
        return label_smoothing_loss_with_logits(pred, target, self.smoothing, self.weight, self.ignore_index, self.reduction)
