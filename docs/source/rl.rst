Reinforcement Learning
======================
This module is a collection of common RL approaches implemented in Lightning.

---------

Module authors
--------------

Contributions by: `Donal Byrne <https://github.com/djbyrne>`_

- DQN
- Double DQN
- Dueling DQN
- Noisy DQN
- Prioritized Experience Replay DQN
- NStep DQN
- Noisy DQN
- Reinforce
- Policy Gradient

------------

DQN Models
----------
The following models are based on DQN

Deep-Q-Network (DQN)
^^^^^^^^^^^^^^^^^^^^
DQN model introduced in `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_.
Paper authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller.

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_
Example::

    from pl_bolts.models.rl import DQN

    model = DQN()
    trainer = Trainer()
    trainer.fit(model)

.. autoclass:: pl_bolts.models.rl.DQN
   :noindex:

