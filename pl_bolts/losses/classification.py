import torch
from torch import nn


class FocalLoss(nn.Module):
    """
    Computes the focal loss

    Example:

        >>> pred = torch.tensor([0.3716, -0.4927, -0.4424, -2.5253 , 0.4039])
        >>> target = torch.tensor([0, 1, 1, 1, 0])
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.)
        >>> loss = loss_fn(pred, target)
        >>> loss
        tensor(0.2441)

    """

    def __init__(
            self,
            alpha = 0.25,
            gamma = 2.
    ):
        """
        Args:
            alpha: balancing factor
            gamma: modulating factor
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Actual loss computation

        Args:
            pred: predicted logits
            target: ground truth labels

        Return:
            A Tensor with the focal loss.
        """
        alpha = target * self.alpha + (1 - target) * (1 - self.alpha)
        probs = pred.sigmoid()
        pt = torch.where(target == 1, probs, 1 - probs)
        ce = -torch.log(pt)
        loss = alpha * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Computes the label-smoothing loss

    Example:

        >>> pred = torch.tensor([[-0.0721,  0.0701], [ 0.8741,  0.7051], [ 0.1122, -0.4712]])
        >>> target = torch.tensor([0, 1, 1])
        >>> loss_fn = LabelSmoothingLoss(smoothing=0.1)
        >>> loss = loss_fn(pred, target)
        >>> loss
        tensor(0.8284)

    """

    def __init__(
            self,
            smoothing=0.1,
    ):
        """
        Args:
            smoothing: smoothing factor
        """
        super().__init__()

        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Actual loss computation

        Args:
            pred: predicted logits
            target: ground truth labels

        Return:
            A Tensor with the label-smoothing loss.
        """
        with torch.no_grad():
            one_hot = torch.zeros_like(pred)
            one_hot.fill_(self.smoothing / (pred.size(1) - 1))
            one_hot.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)

        pred = torch.log_softmax(pred, dim=1)
        loss = (-one_hot * pred).sum(dim=1).mean()
        return loss
