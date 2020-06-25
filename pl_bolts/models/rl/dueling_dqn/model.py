"""
Dueling Deep Q-network
"""


from pl_bolts.models.rl.common.networks import DuelingCNN
from pl_bolts.models.rl.dqn.model import DQN


class DuelingDQN(DQN):
    """
        PyTorch Lightning implementation of `Dueling DQN <https://arxiv.org/abs/1511.06581>`_

        Paper authors: Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas

        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:

            >>> from pl_bolts.models.rl.dueling_dqn.model import DuelingDQN
            ...
            >>> model = DQN("PongNoFrameskip-v4")

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
        """

    def build_networks(self) -> None:
        """Initializes the Dueling DQN train and target networks"""
        self.net = DuelingCNN(self.obs_shape, self.n_actions)
        self.target_net = DuelingCNN(self.obs_shape, self.n_actions)
