Classic ML Models
=================
This module implements classic machine learning models in PyTorch Lightning, including linear regression and logistic
regression. Unlike other libraries that implement these models, here we use PyTorch to enable multi-GPU, multi-TPU and
half-precision training.

---------------

Linear Regression
-----------------
Linear regression fits a linear model between a real-valued target variable ($y$) and one or more features ($X$). We
estimate the regression coefficients $\beta$ that minimizes the mean squared error between the predicted and true target
values.

.. code-block:: python

    from pl_bolts.models import LinearRegression
    import pytorch_lightning as pl
    from pl_bolts.datamodules import SklearnDataModule
    from sklearn.datasets import load_boston

    X, y = load_boston(return_X_y=True)
    loaders = SklearnDataModule(X, y)

    model = LinearRegression(input_dim=13)
    trainer = pl.Trainer()
    trainer.fit(model, loaders.train_dataloader(), loaders.val_dataloader())
    trainer.test(test_dataloaders=loaders.test_dataloader())

.. autoclass:: pl_bolts.models.regression.linear_regression.LinearRegression
   :noindex:

Logistic Regression
-------------------
Logistic regression is a linear model used for classification, i.e. when we have a categorical target variable.
This implementation supports both binary classification (when the number of classes/labels is 2) as well as multi-class
classification. We can think of logistic regression as a one-layer neural network with a logistic (or sigmoid) function
as the activation function in the case of binary classification. In the case of multi-class classification, we can use
a generalization of the One-vs-All approach by outputting probabilities for each class and returning the "argmax", the
class label corresponding to the highest probability.

.. code-block:: python

    from pl_bolts.models import LogisticRegression
    import pytorch_lightning as pl
    from pl_bolts.datamodules import SklearnDataLoaders
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    loaders = SklearnDataLoaders(X, y)

    model = LogisticRegression(input_dim=4, num_classes=3)
    trainer = pl.Trainer()
    trainer.fit(model, loaders.train_dataloader(), loaders.val_dataloader())
    trainer.test(test_dataloaders=loaders.test_dataloader(batch_size=12))

.. autoclass:: pl_bolts.models.regression.logistic_regression.LogisticRegression
   :noindex:
