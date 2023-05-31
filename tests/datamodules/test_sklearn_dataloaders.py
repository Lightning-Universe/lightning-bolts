from warnings import warn

import numpy as np
from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
from pytorch_lightning import seed_everything

try:
    from sklearn.utils import shuffle as sk_shuffle
except ImportError:
    warn(  # pragma: no-cover
        "You want to use `sklearn` which is not installed yet, install it with `pip install sklearn`."
    )


def test_dataloader():
    seed_everything()

    X = np.random.rand(5, 2)
    y = np.random.rand(5)
    x_val = np.random.rand(2, 2)
    y_val = np.random.rand(2)
    x_test = np.random.rand(1, 2)
    y_test = np.random.rand(1)

    shuffled_X, shuffled_y = sk_shuffle(X, y, random_state=1234)

    # -----------------------------
    # train
    # -----------------------------
    loaders = SklearnDataModule(X=X, y=y, val_split=0.2, test_split=0.2, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_X[2:] == train_loader.dataset.X)
    assert np.all(shuffled_X[0] == val_loader.dataset.X)
    assert np.all(shuffled_X[1] == test_loader.dataset.X)
    assert np.all(shuffled_y[2:] == train_loader.dataset.Y)

    # -----------------------------
    # train + val
    # -----------------------------
    loaders = SklearnDataModule(X=X, y=y, x_val=x_val, y_val=y_val, test_split=0.2, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_X[1:] == train_loader.dataset.X)
    assert np.all(x_val == val_loader.dataset.X)
    assert np.all(shuffled_X[0] == test_loader.dataset.X)

    # -----------------------------
    # train + test
    # -----------------------------
    loaders = SklearnDataModule(
        X=X, y=y, x_test=x_test, y_test=y_test, val_split=0.2, random_state=1234, drop_last=True
    )
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_X[1:] == train_loader.dataset.X)
    assert np.all(shuffled_X[0] == val_loader.dataset.X)
    assert np.all(x_test == test_loader.dataset.X)

    # -----------------------------
    # train + val + test
    # -----------------------------
    loaders = SklearnDataModule(X, y, x_val, y_val, x_test, y_test, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(shuffled_X == train_loader.dataset.X)
    assert np.all(x_val == val_loader.dataset.X)
    assert np.all(x_test == test_loader.dataset.X)
