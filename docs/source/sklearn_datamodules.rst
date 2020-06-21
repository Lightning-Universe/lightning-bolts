.. role:: hidden
    :class: hidden-section

Sklearn Datamodules
===================
Utilities to map sklearn or numpy datasets to PyTorch Dataloaders with automatic data splits and GPU/TPU support.


Example:

    >>> from sklearn.datasets import load_boston
    >>> from pl_bolts.datamodules import SklearnDataset
    ...
    >>> X, y = load_boston(return_X_y=True)
    >>> dataset = SklearnDataset(X, y)
    506

Sklearn Dataset
---------------
Transforms a sklearn or numpy dataset to a PyTorch Dataset.

.. autoclass:: pl_bolts.datamodules.sklearn_dataloaders.SklearnDataset
   :noindex:

Sklearn DataLoaders
-------------------
Automatically generates the train, validation and test splits for a Numpy dataset.
They are set up as dataloaders for convenience. Optionally, you can pass in your own validation and test splits.

.. autoclass:: pl_bolts.datamodules.sklearn_dataloaders.SklearnDataLoaders
   :noindex: