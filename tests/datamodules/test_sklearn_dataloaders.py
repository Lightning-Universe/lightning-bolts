from warnings import warn

import numpy as np
import pytest
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
from pytorch_lightning import seed_everything

try:
    from sklearn.utils import shuffle as sk_shuffle

    _SKLEARN_AVAILABLE = True
except ImportError:
    warn("Failing to import `sklearn` correctly")
    _SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="failing to import SKLearn")
def test_dataloader():
    seed_everything()

    x = np.random.rand(5, 2)
    y = np.random.rand(5)
    x_val = np.random.rand(2, 2)
    y_val = np.random.rand(2)
    x_test = np.random.rand(1, 2)
    y_test = np.random.rand(1)

    shuffled_x, shuffled_y = sk_shuffle(x, y, random_state=1234)

    # -----------------------------
    # train
    # -----------------------------
    loaders = SklearnDataModule(X=x, y=y, val_split=0.2, test_split=0.2, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_x[2:] == train_loader.dataset.data)
    assert np.all(shuffled_x[0] == val_loader.dataset.data)
    assert np.all(shuffled_x[1] == test_loader.dataset.data)
    assert np.all(shuffled_y[2:] == train_loader.dataset.labels)

    # -----------------------------
    # train + val
    # -----------------------------
    loaders = SklearnDataModule(X=x, y=y, x_val=x_val, y_val=y_val, test_split=0.2, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_x[1:] == train_loader.dataset.data)
    assert np.all(x_val == val_loader.dataset.data)
    assert np.all(shuffled_x[0] == test_loader.dataset.data)

    # -----------------------------
    # train + test
    # -----------------------------
    loaders = SklearnDataModule(
        X=x, y=y, x_test=x_test, y_test=y_test, val_split=0.2, random_state=1234, drop_last=True
    )
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_x[1:] == train_loader.dataset.data)
    assert np.all(shuffled_x[0] == val_loader.dataset.data)
    assert np.all(x_test == test_loader.dataset.data)

    # -----------------------------
    # train + val + test
    # -----------------------------
    loaders = SklearnDataModule(x, y, x_val, y_val, x_test, y_test, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_x == train_loader.dataset.data)
    assert np.all(x_val == val_loader.dataset.data)
    assert np.all(x_test == test_loader.dataset.data)
