"""
Dueling DQN
===========

The Q value that we are trying to approximate can be divided into two parts, the value state V(s) and the 'advantage'
 of actions in that state A(s, a). Instead of having one full network estimate the entire Q value, Dueling DQN uses two
 estimator heads in order to separate the estimation of the two parts.

The value is the same as in value iteration. It is the discounted expected reward achieved from state s. Think of the
 value as the 'base reward' from being in state s.

The advantage tells us how much 'extra' reward we get from taking action a while in state s. The advantage bridges the
 gap between Q(s, a) and V(s) as Q(s, a) = V(s) + A(s, a).

In the paper [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) the
 network uses two heads, one outputs the value state and the other outputs the advantage. This leads to better
 training stability, faster convergence and overall better results. The V head outputs a single scalar
 (the state value), while the advantage head outputs a tensor equal to the size of the action space, containing
 an advantage value for each action in state s.

Changing the network architecture is not enough, we also need to ensure that the advantage mean is 0. This is done
 by subtracting the mean advantage from the Q value. This essentially pulls the mean advantage to 0.

.. math::

    Q(s, a) = V(s) + A(s, a) - 1/N * sum_k(A(s, k)

Benefits
--------

- Ability to efficiently learn the state value function. In the dueling network, every Q update also updates the Value
    stream, where as in DQN only the value of the chosen action is updated. This provides a better approximation of the
    values
- The differences between total Q values for a given state are quite small in relation to the magnitude of Q. The
    difference in the Q values between the best action and the second best action can be very small, while the average
    state value can be much larger. The differences in scale can introduce noise, which may lead to the greedy policy
    switching the priority of these actions. The seperate estimators for state value and advantage makes the Dueling
    DQN robust to this type of scenario

In order to update the basic DQN to a Dueling DQN we need to do the following

Add Network Heads
^^^^^^^^^^^^^^^^^

.. code-block:: python

    #  add this to the dqn network
    conv_out_size = self._get_conv_out(input_shape)

    # advantage head
    self.fc_adv = nn.Sequential(
        nn.Linear(conv_out_size, 256),
        nn.ReLU(),
        nn.Linear(256, n_actions)
    )

    # value head
    self.fc_val = nn.Sequential(
        nn.Linear(conv_out_size, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )


Update Forward
^^^^^^^^^^^^^^

.. code-block:: python

    def forward(self, x):
        adv, val = self.adv_val(x)

        # return the full Q value which is value + adv while we pull the mean to 0
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256 # normalize
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)


Results
-------

The results below a noticeable improvement from the original DQN network.

Pong
----

Dueling DQN baseline
^^^^^^^^^^^^^^^^^^^^

Similar to the results of the DQN baseline, the agent has a period where the number of steps per episodes increase as
 it begins to hold its own against the heuristic oppoent, but then the steps per episode quickly begins to drop
 as it gets better and starts to beat its opponent faster and faster. There is a noticable point at step ~250k
 where the agent goes from losing to winning.

As you can see by the total rewards, the dueling network's training progression is very stable and continues to trend
 upward until it finally plateus.

![Dueling DQN Results](../../docs/images/pong_dueling_dqn_results.png)

DQN vs Dueling DQN
^^^^^^^^^^^^^^^^^^

In comparison to the base DQN, we see that the Dueling network's training is much more stable and is able to reach a
 score in the high teens faster than the DQN agent. Even though the Dueling network is more stable and out performs DQN
 early in training, by the end of training the two networks end up at the same point.

This could very well be due to the simplicity of the Pong environment.

 - Orange: DQN
 - Red: Dueling DQN

![Dueling DQN Results](../../docs/images/pong_dueling_dqn_comparison.png)


"""


from pl_bolts.models.rl.common.networks import DuelingCNN
from pl_bolts.models.rl.dqn_model import DQN


class DuelingDQN(DQN):
    """
        PyTorch Lightning implementation of `Dueling DQN <https://arxiv.org/abs/1511.06581>`_

        Paper authors: Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas

        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:

            >>> from pl_bolts.models.rl.dueling_dqn_model import DuelingDQN
            ...
            >>> model = DuelingDQN("PongNoFrameskip-v4")

        Train::

            trainer = Trainer()
            trainer.fit(model)

        Args:
            env: gym environment tag
            gpus: number of gpus being used
            eps_start: starting value of epsilon for the epsilon-greedy exploration
            eps_end: final value of epsilon for the epsilon-greedy exploration
            eps_last_frame: the final frame in for the decrease of epsilon. At this frame espilon = eps_end
            sync_rate: the number of iterations between syncing up the target network with the train network
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            replay_size: total capacity of the replay buffer
            warm_start_size: how many random steps through the environment to be carried out at the start of
                training to fill the buffer with a starting point
            sample_len: the number of samples to pull from the dataset iterator and feed to the DataLoader

        .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

        """

    def build_networks(self) -> None:
        """Initializes the Dueling DQN train and target networks"""
        self.net = DuelingCNN(self.obs_shape, self.n_actions)
        self.target_net = DuelingCNN(self.obs_shape, self.n_actions)
