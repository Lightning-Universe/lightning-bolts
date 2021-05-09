.. role:: hidden
    :class: hidden-section

Self-supervised Callbacks
=========================
Useful callbacks for self-supervised learning models

---------------

BYOLMAWeightUpdate
------------------
The exponential moving average weight-update rule from Bootstrap Your Own Latent (BYOL).

.. autoclass:: pl_bolts.callbacks.byol_updates.BYOLMAWeightUpdate
   :noindex:

----------------

SSLOnlineEvaluator
------------------
Appends a MLP for fine-tuning to the given model. Callback has its own mini-inner loop.

.. autoclass:: pl_bolts.callbacks.ssl_online.SSLOnlineEvaluator
   :noindex:
