*******
Loggers
*******

Lightning supports the most popular logging frameworks (TensorBoard, Comet, etc...). TensorBoard is used by default, 
but you can pass to the :class:`~pytorch_lightning.trainer.trainer.Trainer` any combintation of the following loggers.

.. note::

    All loggers log by default to `os.getcwd()`. To change the path without creating a logger set
    `Trainer(default_root_dir='/your/path/to/save/checkpoints')`

Read more about :ref:`logging` options.

Azure Machine Learning
======================

`Azure Machine Learning<https://docs.microsoft.com/en-us/azure/machine-learning/>`_ is a third-party logger.
It uses the `Azure Machine Learning + MLFlow<https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow>`_ integration under the hood.
To use :class:`~pytorch_lightning_bolts.loggers.AzureMlLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install azureml-mlflow

Then construct the logger and pass it to the `Trainer`:

.. testcode::

    
    from pytorch_lightning.loggers import AzureMlLogger

    azureml_logger = AzureMlLogger()
    trainer = Trainer(max_epochs=10, logger=azureml_logger)
