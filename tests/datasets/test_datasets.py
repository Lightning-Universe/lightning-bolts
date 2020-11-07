from torch.utils.data import DataLoader

from pl_bolts.datasets import DummyDataset, RandomDataset, RandomDictDataset, RandomDictStringDataset


def test_dummy_ds():
    ds = DummyDataset((1, 2), num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_ds():
    ds = RandomDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_dict_ds():
    ds = RandomDictDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_str_dict_ds():
    ds = RandomDictStringDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass
