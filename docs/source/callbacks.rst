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

Info callbacks
--------------
These callbacks give all sorts of useful information during training.


PrintTableMetricsCallback
^^^^^^^^^^^^^^^^^^^^^^^^^
This callbacks prints training metrics to a table.
It's very bare-bones for speed purposes.

.. autoclass:: pl_bolts.callbacks.printing.PrintTableMetricsCallback
   :noindex:
