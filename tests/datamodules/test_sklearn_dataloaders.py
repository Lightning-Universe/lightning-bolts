from warnings import warn

import numpy as np
from pytorch_lightning import seed_everything

from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule

try:
    from sklearn.utils import shuffle as sk_shuffle
except ImportError:
    warn(  # pragma: no-cover
        'You want to use `sklearn` which is not installed yet, install it with `pip install sklearn`.'
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
    assert np.all(train_loader.dataset.X == shuffled_X[2:])
    assert np.all(val_loader.dataset.X == shuffled_X[0])
    assert np.all(test_loader.dataset.X == shuffled_X[1])
    assert np.all(train_loader.dataset.Y == shuffled_y[2:])

    # -----------------------------
    # train + val
    # -----------------------------
    loaders = SklearnDataModule(X=X, y=y, x_val=x_val, y_val=y_val, test_split=0.2, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(train_loader.dataset.X == shuffled_X[1:])
    assert np.all(val_loader.dataset.X == x_val)
    assert np.all(test_loader.dataset.X == shuffled_X[0])

    # -----------------------------
    # train + test
    # -----------------------------
    loaders = SklearnDataModule(
        X=X, y=y, x_test=x_test, y_test=y_test, val_split=0.2, random_state=1234, drop_last=True
    )
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(train_loader.dataset.X == shuffled_X[1:])
    assert np.all(val_loader.dataset.X == shuffled_X[0])
    assert np.all(test_loader.dataset.X == x_test)

    # -----------------------------
    # train + val + test
    # -----------------------------
    loaders = SklearnDataModule(X, y, x_val, y_val, x_test, y_test, random_state=1234, drop_last=True)
    train_loader = loaders.train_dataloader()
    val_loader = loaders.val_dataloader()
    test_loader = loaders.test_dataloader()
    assert np.all(train_loader.dataset.X == shuffled_X)
    assert np.all(val_loader.dataset.X == x_val)
    assert np.all(test_loader.dataset.X == x_test)
