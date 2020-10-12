from pl_bolts.datasets import DummyDataset, RandomDataset, RandomDictDataset, RandomDictStringDataset
from torch.utils.data import DataLoader


def test_dummy_ds(tmpdir):
    ds = DummyDataset((1, 2), num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_ds(tmpdir):
    ds = RandomDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_dict_ds(tmpdir):
    ds = RandomDictDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_str_dict_ds(tmpdir):
    ds = RandomDictStringDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass
