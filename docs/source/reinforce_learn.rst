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
- NStep DQN
- Prioritized Experience Replay DQN
- Reinforce
- Vanilla Policy Gradient

------------

.. note:: 
          RL models currently only support CPU and single GPU training with `distributed_backend=dp`. Full GPU
          support will be added in later updates.


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
    dqn = DQN("PongNoFrameskip-v4")
    trainer = Trainer()
    trainer.fit(dqn)

.. autoclass:: pl_bolts.models.rl.dqn_model.DQN
   :noindex:

Double DQN
^^^^^^^^^^^^^^^^^^^^
Double DQN model introduced in `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/pdf/1509.06461.pdf>`_
Paper authors: Hado van Hasselt, Arthur Guez, David Silver

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_

Example::

    from pl_bolts.models.rl import DoubleDQN
    ddqn = DoubleDQN("PongNoFrameskip-v4")
    trainer = Trainer()
    trainer.fit(ddqn)

.. autoclass:: pl_bolts.models.rl.double_dqn_model.DoubleDQN
   :noindex:

Dueling DQN
^^^^^^^^^^^^^^^^^^^^
Dueling DQN model introduced in `Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`_
Paper authors: Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_

Example::

    from pl_bolts.models.rl import DuelingDQN
    dueling_dqn = DuelingDQN("PongNoFrameskip-v4")
    trainer = Trainer()
    trainer.fit(dueling_dqn)

.. autoclass:: pl_bolts.models.rl.dueling_dqn_model.DuelingDQN
   :noindex:

Noisy DQN
^^^^^^^^^^^^^^^^^^^^
Noisy DQN model introduced in `Noisy Networks for Exploration <https://arxiv.org/abs/1706.10295>`_
Paper authors: Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves,
Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, Shane Legg

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_

Example::

    from pl_bolts.models.rl import NoisyDQN
    noisy_dqn = NoisyDQN("PongNoFrameskip-v4")
    trainer = Trainer()
    trainer.fit(noisy_dqn)

.. autoclass:: pl_bolts.models.rl.noisy_dqn_model.NoisyDQN
   :noindex:


N-Step DQN
^^^^^^^^^^^^^^^^^^^^
N-Step DQN model introduced in `Learning to Predict by the Methods of Temporal Differences  <http://incompleteideas.net/papers/sutton-88-with-erratum.pdf>`_
Paper authors: Richard S. Sutton

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_

Example::

    from pl_bolts.models.rl import NStepDQN
    n_step_dqn = NStepDQN("PongNoFrameskip-v4")
    trainer = Trainer()
    trainer.fit(n_step_dqn)

.. autoclass:: pl_bolts.models.rl.n_step_dqn_model.NStepDQN
   :noindex:


Prioritized Experience Replay DQN
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Double DQN model introduced in `Prioritized Experience Replay  <http://incompleteideas.net/papers/sutton-88-with-erratum.pdf>`_
Paper authors: Tom Schaul, John Quan, Ioannis Antonoglou, David Silver

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_

Example::

    from pl_bolts.models.rl import PERDQN
    per_dqn = PERDQN("PongNoFrameskip-v4")
    trainer = Trainer()
    trainer.fit(per_dqn)

.. autoclass:: pl_bolts.models.rl.per_dqn_model.PERDQN
   :noindex:



--------------

Policy Gradient Models
----------------------
The following models are based on Policy gradient

REINFORCE
^^^^^^^^^^^^^^^^^^^^
REINFORCE model introduced in `Policy Gradient Methods For Reinforcement Learning With Function Approximation <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_

Example::

    from pl_bolts.models.rl import Reinforce
    reinforce = Reinforce("CartPole-v0")
    trainer = Trainer()
    trainer.fit(reinforce)

.. autoclass:: pl_bolts.models.rl.reinforce_model.Reinforce
   :noindex:


Vanilla Policy Gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Vanilla Policy Gradient model introduced in `Policy Gradient Methods For Reinforcement Learning With Function Approximation <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour

Original implementation by: `Donal Byrne <https://github.com/djbyrne>`_

Example::

    from pl_bolts.models.rl import PolicyGradient
    vpg = PolicyGradient("CartPole-v0")
    trainer = Trainer()
    trainer.fit(vpg)

.. autoclass:: pl_bolts.models.rl.vanilla_policy_gradient_model.PolicyGradient
   :noindex:
