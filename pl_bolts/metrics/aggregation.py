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


def precision_at_k(output, target, top_k=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
