Classic ML Models
=================
This module implements classic machine learning models in PyTorch Lightning, including linear regression and logistic
regression. Unlike other libraries that implement these models, here we use PyTorch to enable multi-GPU, multi-TPU and
half-precision training.

---------------

Linear Regression
-----------------
Linear regression fits a linear model between a real-valued target variable :math:`y` and one or more features :math:`X`. We
estimate the regression coefficients that minimize the mean squared error between the predicted and true target
values.

We formulate the linear regression model as a single-layer neural network. By default we include only one neuron in
the output layer, although you can specify the `output_dim` yourself.

Add either L1 or L2 regularization, or both, by specifying the regularization strength (default 0).

.. code-block:: python

    from pl_bolts.models.regression import LinearRegression
    import pytorch_lightning as pl
    from pl_bolts.datamodules import SklearnDataModule
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    loaders = SklearnDataModule(X, y)

    model = LinearRegression(input_dim=13)
    trainer = pl.Trainer()
    trainer.fit(model, train_dataloader=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())
    trainer.test(test_dataloaders=loaders.test_dataloader())

.. autoclass:: pl_bolts.models.regression.linear_regression.LinearRegression
   :noindex:

-------------

Logistic Regression
-------------------
Logistic regression is a linear model used for classification, i.e. when we have a categorical target variable.
This implementation supports both binary and multi-class classification.

In the binary case, we formulate the logistic regression model as a one-layer neural network with one neuron in the
output layer and a sigmoid activation function. In the multi-class case, we use a single-layer neural network but now
with :math:`k` neurons in the output, where :math:`k` is the number of classes. This is also referred to as multinomial
logistic regression.

Add either L1 or L2 regularization, or both, by specifying the regularization strength (default 0).

.. code-block:: python

    from sklearn.datasets import load_iris
    from pl_bolts.models.regression import LogisticRegression
    from pl_bolts.datamodules import SklearnDataModule
    import pytorch_lightning as pl

    # use any numpy or sklearn dataset
    X, y = load_iris(return_X_y=True)
    dm = SklearnDataModule(X, y)

    # build model
    model = LogisticRegression(input_dim=4, num_classes=3)

    # fit
    trainer = pl.Trainer(tpu_cores=8, precision=16)
    trainer.fit(model, train_dataloader=dm.train_dataloader(), val_dataloaders=dm.val_dataloader())

    trainer.test(test_dataloaders=dm.test_dataloader(batch_size=12))

Any input will be flattened across all dimensions except the first one (batch).
This means images, sound, etc... work out of the box.

.. code-block:: python

    # create dataset
    dm = MNISTDataModule(num_workers=0, data_dir=tmpdir)

    model = LogisticRegression(input_dim=28 * 28, num_classes=10, learning_rate=0.001)
    model.prepare_data = dm.prepare_data
    model.train_dataloader = dm.train_dataloader
    model.val_dataloader = dm.val_dataloader
    model.test_dataloader = dm.test_dataloader

    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model)
    trainer.test(model)
    # {test_acc: 0.92}

.. autoclass:: pl_bolts.models.regression.logistic_regression.LogisticRegression
   :noindex:
