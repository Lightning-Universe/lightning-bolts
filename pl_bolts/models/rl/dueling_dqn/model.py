"""
Dueling Deep Q-network
"""


from pl_bolts.models.rl.common.networks import DuelingCNN
from pl_bolts.models.rl.dqn.model import DQN


class DuelingDQN(DQN):
    """ Dueling DQN Model """

    def build_networks(self) -> None:
        """Initializes the Dueling DQN train and target networks"""
        self.net = DuelingCNN(self.obs_shape, self.n_actions)
        self.target_net = DuelingCNN(self.obs_shape, self.n_actions)
