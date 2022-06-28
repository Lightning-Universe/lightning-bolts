Semi-supervised learning
========================
Collection of utilities for semi-supervised learning where some part of the data is labeled
and the other part is not.

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

--------------

Balanced classes
----------------

Example::

    from pl_bolts.utils.semi_supervised import balance_classes


.. autofunction:: pl_bolts.utils.semi_supervised.balance_classes
    :noindex:

half labeled batches
--------------------

Example::

    from pl_bolts.utils.semi_supervised import balance_classes

.. autofunction:: pl_bolts.utils.semi_supervised.generate_half_labeled_batches
    :noindex:
