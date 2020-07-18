import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(pred, target)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1 - pt) ** self.gamma) * CE_loss
        return F_loss.mean()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0., dim=-1):
        super().__init__()
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        with torch.no_grad():
            one_hot = torch.zeros_like(pred)
            one_hot.fill_(self.smoothing / (pred.size(-1) - 1))
            one_hot.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)

        return torch.mean(torch.sum(-one_hot * pred.log_softmax(dim=self.dim), dim=self.dim))
