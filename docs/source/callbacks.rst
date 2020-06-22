.. role:: hidden
    :class: hidden-section

Bolts Callbacks
===============
This module houses a collection of callbacks that can be passed into the trainer

Example::

from pl_bolts.callbacks import PrintTableMetricsCallback
import pytorch_lightning as pl

trainer = pl.Trainer(callbacks=[PrintTableMetricsCallback()])


.. automodule:: pl_bolts.callbacks
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
