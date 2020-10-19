from pl_bolts.datasets.cifar10_dataset import CIFAR10


def test_dev_datasets(tmpdir):

    ds = CIFAR10(tmpdir)
    for b in ds:
        pass
