.. role:: hidden
    :class: hidden-section

Sklearn Datamodule
==================
Utilities to map sklearn or numpy datasets to PyTorch Dataloaders with automatic data splits and GPU/TPU support.

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from pl_bolts.datamodules import SklearnDataModule

    X, y = load_diabetes(return_X_y=True)
    loaders = SklearnDataModule(X, y)

    train_loader = loaders.train_dataloader(batch_size=32)
    val_loader = loaders.val_dataloader(batch_size=32)
    test_loader = loaders.test_dataloader(batch_size=32)

Or build your own torch datasets

.. code-block:: python

    from sklearn.datasets import load_diabetes
    from pl_bolts.datamodules import SklearnDataset

    X, y = load_diabetes(return_X_y=True)
    dataset = SklearnDataset(X, y)
    loader = DataLoader(dataset)

----------------

Sklearn Dataset Class
---------------------
Transforms a sklearn or numpy dataset to a PyTorch Dataset.

.. autoclass:: pl_bolts.datamodules.sklearn_datamodule.SklearnDataset
   :noindex:

----------------

Sklearn DataModule Class
------------------------
Automatically generates the train, validation and test splits for a Numpy dataset.
They are set up as dataloaders for convenience. Optionally, you can pass in your own validation and test splits.

.. autoclass:: pl_bolts.datamodules.sklearn_datamodule.SklearnDataModule
