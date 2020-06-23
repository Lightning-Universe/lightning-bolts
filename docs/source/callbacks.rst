.. role:: hidden
    :class: hidden-section

Bolts Callbacks
===============
This module houses a collection of callbacks that can be passed into the trainer

.. code-block:: python

    from pl_bolts.callbacks import PrintTableMetricsCallback
    import pytorch_lightning as pl

    trainer = pl.Trainer(callbacks=[PrintTableMetricsCallback()])

    # loss│train_loss│val_loss│epoch
    # ──────────────────────────────
    # 2.2541470527648926│2.2541470527648926│2.2158432006835938│0

---------------

Create a Callback
-----------------
Creating a callback is simple:

.. code-block:: python

    from pytorch_lightning.callbacks import Callback

    class MyCallback(Callback)
        def on_epoch_end(self, trainer, pl_module):
            # do something

Please refer to `Callback docs <https://pytorch-lightning.readthedocs.io/en/stable/callbacks.html#callback-base>`_
for a full list of the 20+ hooks available.
