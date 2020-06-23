AMDIM Tasks
===========
The following are pretext tasks used in the style of AMDIM.
In general the AMDIM pretext task uses triplets of features, the positive and anchor features
come from different augmentations of the same image, while the negative features come from another image

.. code-block:: python

    x_pos = data_augmentation(x1)
    x_anchor = data_augmentation(x1)
    x_negative = data_augmentation(x2)

The AMDIM taks compares feature maps from different layers of an encoder applied to each input

.. code-block:: python

    # f1, g1 are feature maps (batch, c, 1, 1)
    # f5, g5 are feature maps (batch, c, 5, 5), etc...
    (f1, f5, f7) = encoder(x_pos)
    (g1, g5, g7) = encoder(r_anchor)

Each individual AMDIM task then combines each of these feature maps in a particular way.
below, we implement the original task, along with a few derivations.

-------------

AMDIM_15_17_55_ContrastiveTask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass::pl_bolts.losses.self_supervised_learning.AMDIM_15_17_55_ContrastiveTask
    :noindex:


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