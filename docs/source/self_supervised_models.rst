Self-supervised
===============
This bolts module houses a collection of all self-supervised learning models.

Self-supervised learning extracts representations of an input by solving a pretext task. In this package,
we implement many of the current state-of-the-art self-supervised algorithms.

-----------------

Contrastive Learning
--------------------
The following models perform contrastive learning.

AMDIM
^^^^^

.. autoclass:: pl_bolts.models.self_supervised.AMDIM
   :noindex:

CPC (V2)
^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.CPCV2
   :noindex:

SimCLR
^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.SimCLR
   :noindex:

Moco (V2)
^^^^^^^^^

.. autoclass:: pl_bolts.models.self_supervised.MocoV2
   :noindex:
