.. role:: hidden
    :class: hidden-section

Monitoring Callbacks
====================

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


---------------

Model Verification
------------------


Gradient-Check for Batch-Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gradient descent over a batch of samples can not only benefit the optimization but also leverages data parallelism.
However, one has to be careful not to mix data across the batch dimension.
Only a small error in a reshape or permutation operation results in the optimization getting stuck and you won't
even get a runtime error. How can one tell if the model mixes data in the batch?
A simple trick is to do the following:

1. run the model on an example batch (can be random data)
2. get the output batch and select the n-th sample (choose n)
3. compute a dummy loss value of only that sample and compute the gradient w.r.t the entire input batch
4. observe that only the i-th sample in the input batch has non-zero gradient

|

If the gradient is non-zero for the other samples in the batch, it means the forward pass of the model is mixing data!
The :class:`~pl_bolts.callbacks.verification.batch_gradient.BatchGradientVerificationCallback`
does all of that for you before training begins.

.. code-block:: python

    from pytorch_lightning import Trainer
    from pl_bolts.callbacks import BatchGradientVerificationCallback

    model = YourLightningModule()
    verification = BatchGradientVerificationCallback()
    trainer = Trainer(callbacks=[verification])
    trainer.fit(model)

This Callback will warn the user with the following message in case data mixing inside the batch is detected:

.. code-block::

    Your model is mixing data across the batch dimension.
    This can lead to wrong gradient updates in the optimizer.
    Check the operations that reshape and permute tensor dimensions in your model.


A non-Callback version
:class:`~pl_bolts.callbacks.verification.batch_gradient.BatchGradientVerification`
that works with any PyTorch :class:`~torch.nn.Module` is also available:

.. code-block:: python

    from pl_bolts.utils import BatchGradientVerification

    model = YourPyTorchModel()
    verification = BatchGradientVerification(model)
    valid = verification.check(input_array=torch.rand(2, 3, 4), sample_idx=1)

In this example we run the test on a batch size 2 by inspecting gradients on the second sample.
