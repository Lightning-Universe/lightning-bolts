.. role:: hidden
    :class: hidden-section

Info Callbacks
==============

These callbacks give all sorts of useful information during training.

---------------

Print Table Metrics
-------------------
This callback prints training metrics to a table.
It's very bare-bones for speed purposes.

.. autoclass:: pl_bolts.callbacks.printing.PrintTableMetricsCallback
   :noindex:


---------------

Data Monitoring in LightningModule
----------------------------------
The data monitoring callbacks allow you to log and inspect the distribution of data that passes through
the training step and layers of the model. When used in combination with a supported logger, the
:class:`~pl_bolts.callbacks.data_monitor.TrainingDataMonitor` creates a histogram for each `batch` input in
:meth:`~pytorch_lightning.core.lightning.LightningModule.training_step` and sends it to the logger:

.. code-block:: python

    from pl_bolts.callbacks import TrainingDataMonitor
    from pytorch_lightning import Trainer

    # log the histograms of input data sent to LightningModule.training_step
    monitor = TrainingDataMonitor(log_every_n_steps=25)

    model = YourLightningModule()
    trainer = Trainer(callbacks=[monitor])
    trainer.fit()


The second, more advanced :class:`~pl_bolts.callbacks.data_monitor.ModuleDataMonitor`
callback tracks histograms for the data that passes through
the model itself and its submodules, i.e., it tracks all `.forward()` calls and registers the in- and outputs.
You can track all or just a selection of submodules:

.. code-block:: python

    from pl_bolts.callbacks import ModuleDataMonitor
    from pytorch_lightning import Trainer

    # log the in- and output histograms of LightningModule's `forward`
    monitor = ModuleDataMonitor()

    # all submodules in LightningModule
    monitor = ModuleDataMonitor(submodules=True)

    # specific submodules
    monitor = ModuleDataMonitor(submodules=["generator", "generator.conv1"])

    model = YourLightningModule()
    trainer = Trainer(callbacks=[monitor])
    trainer.fit()

This is especially useful for debugging the data flow in complex models and to identify
numerical instabilities.
