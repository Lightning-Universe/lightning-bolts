Vision Tasks
============
Here we present common tasks used in computer vision.

------------------

Self-supervised learning
------------------------
Self-supervised learning in computer vision uses a pretext task to extract learning signal.
Here is a collection of common tasks

AMDIMContrastiveTask
^^^^^^^^^^^^^^^^^^^^
Implementation modified from the `original repo <https://github.com/Philip-Bachman/amdim-public>`_.
This is the contrastive task from AMDIM (`Philip Bachman, R Devon Hjelm, William Buchwalter <https://arxiv.org/abs/1906.00910>`_).

In this task, we take in two sets of feature maps
(in this case from a positive $x^+$ and anchor example $x^a$) $M^a = \{m_1, m_2, ..., m_i\}, M^+ = \{m_1, m_2, ..., m_j\}$.

The task compares feature maps across spatial locations within the network. Here are the supported possibilities:

The `1:1` task:

Corresponds to using the feature maps $m_{i} \in M^a$ vs $m_{j} \in M^+$ in the NCE Loss $\math{cal}(m_i, m_j)$

.. code-block:: python

    xa, x+ = augmentations(x), augmentations(x)
    ra = model(xa)
    r+ = model(x+)

    # 1:1 task (last two feature maps)
    task = AMDIMContrastiveTask('1:1')
    map_i = ra[-1]
    map_j = r+[-1]
    loss = task(map_i, map_j)

The `1:5,1:7,5:5` task:
This is the task used in the AMDIM paper. This task generates three losses.
Corresponds to using the feature maps $m_{i} \in M^a$ vs $m_{j} \in M^+$ in the NCE Loss $\math{cal}(m_i, m_j)$

$\math{cal}_1(m_i, m_{i-1})$
$\math{cal}_2(m_i, m_{i-2})$
$\math{cal}_3(m_{i-1}, m_{i-1})$

.. code-block:: python

    ra = model(xa)
    r+ = model(x+)

    # each model outputs the last feature maps where the dimensions match the numbers specified
    # f1 = (b, c, 1, 1)
    # f5 = (b, c, 5, 5)
    # f6 = (b, c, 7, 7)
    f1, f5, f7 = ra
    f1, f5, f7 = r+

    task = AMDIMContrastiveTask(strategy='1:5,1:7,5:5')
    loss = task(ra, r+)

Other strategies available are: '1:1,5:5,7:7', '1:random'

.. note:: This task can be generalized for any list of tuples. Feel free to submit a PR!

.. note:: The softmax masking in this task can be sped up. Feel free to submit a PR!


.. autoclass:: pl_bolts.losses.self_supervised_learning.AMDIMContrastiveTask
    :noindex:

