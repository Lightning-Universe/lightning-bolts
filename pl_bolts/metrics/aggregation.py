import torch


def mean(res, key):
    # recursive mean for multilevel dicts
    return torch.stack([x[key] if isinstance(x, dict) else mean(x, key) for x in res]).mean()


def accuracy(preds, labels):
    preds = preds.float()
    max_lgt = torch.max(preds, 1)[1]
    num_correct = (max_lgt == labels).sum().item()
    num_correct = torch.tensor(num_correct).float()
    acc = num_correct / len(labels)

    return acc
