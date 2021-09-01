.. role:: hidden
    :class: hidden-section

Build a Callback
================
This module houses a collection of callbacks that can be passed into the trainer

.. code-block:: python

    from pl_bolts.callbacks import PrintTableMetricsCallback

    trainer = pl.Trainer(callbacks=[PrintTableMetricsCallback()])

    # loss│train_loss│val_loss│epoch
    # ──────────────────────────────
    # 2.2541470527648926│2.2541470527648926│2.2158432006835938│0

------------------

What is a Callback
------------------
A callback is a self-contained program that can be intertwined into a training pipeline without polluting the main
research logic.

---------------

Create a Callback
-----------------
Creating a callback is simple:

.. code-block:: python

    from pytorch_lightning.callbacks import Callback

    class MyCallback(Callback)
        def on_epoch_end(self, trainer, pl_module):
            # do something

Please refer to `Callback docs <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#built-in-callbacks>`_
for a full list of the 20+ hooks available.
