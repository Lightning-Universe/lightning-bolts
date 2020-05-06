import torch


def mean(res, key):
    # recursive mean for multilevel dicts
    return torch.stack([x[key] if isinstance(x, dict) else mean(x, key) for x in res]).mean()
