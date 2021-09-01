Model quality control
=====================
For bolts to be added to the library we have a **rigorous** quality control checklist

Bolts vs my own repo
--------------------
We hope you keep your own repo still! We want to link to it to let people know. However,
by adding your contribution to bolts you get these **additional** benefits!

    1. More visibility! (more people all over the world use your code)
    2. We test your code on every PR (CPUs, GPUs, TPUs).
    3. We host the docs (and test on every PR).
    4. We help you build thorough, beautiful documentation.
    5. We help you build robust tests.
    6. We'll pretrain expensive models for you and host weights.
    7. We will improve the speed of your models!
    8. Eligible for invited talks to discuss your implementation.
    9. Lightning swag + involvement in the broader contributor community :)

.. note:: You still get to keep your attribution and be recognized for your work!

.. note:: Bolts is a community library built by incredible people like you!

Contribution requirements
-------------------------

Benchmarked
^^^^^^^^^^^
Models have known performance results on common baseline datasets.

Device agnostic
^^^^^^^^^^^^^^^
Models must work on CPUs, GPUs and TPUs without changing code. We help authors with this.

.. code-block:: python

    # bad
    encoder.to(device)

Fast
^^^^
We inspect models for computational inefficiencies and help authors meet the bar.
Granted, sometimes the approaches are slow for mathematical reasons. But anything related to engineering we
help overcome.

.. code-block:: python

    # bad
    mtx = ...
    for xi in rows:
        for yi in cols
            mxt[xi, yi] = ...

    # good
    x = x.item().numpy()
    x = np.some_fx(x)
    x = torch.tensor(x)

Tested
^^^^^^
Models are tested on every PR (on CPUs, GPUs and soon TPUs).

- `Live build <https://github.com/PyTorchLightning/lightning-bolts/pull/59/checks>`_
- `Tests <https://github.com/PyTorchLightning/lightning-bolts/tree/master/tests>`_

Modular
^^^^^^^
Models are modularized to be extended and reused easily.

.. code-block:: python

    # GOOD!
    class LitVAE(pl.LightningModule):

        def init_prior(self, ...):
            # enable users to override interesting parts of each model

        def init_posterior(self, ...):
            # enable users to override interesting parts of each model

    # BAD
    class LitVAE(pl.LightningModule):

        def __init__(self):
            self.prior = ...
            self.posterior = ...

Attribution
^^^^^^^^^^^
Any models and weights that are contributed are attributed to you as the author(s).

We request that each contribution have:

    - The original paper link
    - The list of paper authors
    - The link to the original paper code (if available)
    - The link to your repo
    - Your name and your team's name as the implementation authors.
    - Your team's affiliation
    - Any generated examples, or result plots.
    - Hyperparameter configurations for the results.

Thank you for all your amazing contributions!

-------------

The bar seems high
------------------
If your model doesn't yet meet this bar, no worries!
Please open the PR and our team of core contributors will help you get there!

---------------

Do you have contribution ideas?
-------------------------------
Yes! Check the Github issues for requests from the Lightning team and the community!
We'll even work with you to finish your implementation! Then we'll help you pretrain it and cover the compute costs
when possible.
