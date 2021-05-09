Self-supervised Learning Contrastive tasks
==========================================
This section implements popular contrastive learning tasks used in self-supervised learning.

---------------

FeatureMapContrastiveTask
-------------------------
This task compares sets of feature maps.

In general the feature map comparison pretext task uses triplets of features.
Here are the abstract steps of comparison.

Generate multiple views of the same image

.. code-block:: python

    x1_view_1 = data_augmentation(x1)
    x1_view_2 = data_augmentation(x1)

Use a different example to generate additional views (usually within the same batch or a pool of candidates)

.. code-block:: python

    x2_view_1 = data_augmentation(x2)
    x2_view_2 = data_augmentation(x2)

Pick 3 views to compare, these are the anchor, positive and negative features

.. code-block:: python

    anchor = x1_view_1
    positive = x1_view_2
    negative = x2_view_1

Generate feature maps for each view

.. code-block:: python

    (a0, a1, a2) = encoder(anchor)
    (p0, p1, p2) = encoder(positive)

Make a comparison for a set of feature maps

.. code-block:: python

    phi = some_score_function()

    # the '01' comparison
    score = phi(a0, p1)

    # and can be bidirectional
    score = phi(p0, a1)

In practice the contrastive task creates a BxB matrix where B is the batch size. The diagonals for set 1 of feature maps
are the anchors, the diagonals of set 2 of the feature maps are the positives, the non-diagonals of set 1 are the
negatives.

.. autoclass:: pl_bolts.losses.self_supervised_learning.FeatureMapContrastiveTask
    :noindex:

--------------

Context prediction tasks
------------------------
The following tasks aim to predict a target using a context representation.

CPCContrastiveTask
^^^^^^^^^^^^^^^^^^
This is the predictive task from CPC (v2).

.. code-block::

    task = CPCTask(num_input_channels=32)

    # (batch, channels, rows, cols)
    # this should be thought of as 49 feature vectors, each with 32 dims
    Z = torch.random.rand(3, 32, 7, 7)

    loss = task(Z)


.. autoclass:: pl_bolts.losses.self_supervised_learning.CPCTask
    :noindex:
