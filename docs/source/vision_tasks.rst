Vision Tasks
============
TODO


Self-supervised learning
------------------------
Self-supervised learning in computer vision uses a pretext task to extract learning signal.
Here is a collection of common tasks

AMDIMContrastiveTask
^^^^^^^^^^^^^^^^^^^^
This is the contrastive task from AMDIM. In this task, we take in two sets of feature maps
(in this case from a positive $x^+$ and anchor example $x^a$) $M^a = \{m_1, m_2, ..., m_i\}, M^+ = \{m_1, m_2, ..., m_j\}$.

The task compares feature maps across spatial locations within the network. Here are the supported possibilities:

The `1:1` task:

Corresponds $m_{i} \in M^a$ vs $m_{j} \in M^+$

.. code-block:: python

    xa, x+ = augmentations(x), augmentations(x)
    ra = model(xa)
    r+ = model(x+)

    # 1:1 task (last two feature maps)
    task = AMDIMContrastiveTask('1:1')
    map_i = ra[-1]
    map_j = r+[-1]
    loss = task(map_i, map_j)


.. autoclass:: pl_bolts.losses.self_supervised_learning.AmdimNCELoss
    :noindex: