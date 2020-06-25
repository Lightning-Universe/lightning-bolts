"""
N Step Deep Q-network
"""
import torch

from pl_bolts.models.reinforcement.common.experience import NStepExperienceSource
from pl_bolts.models.reinforcement.dqn.model import DQN


class NStepDQN(DQN):
    """ NStep DQN Model """

    def __init__(
            self,
            env: str,
            gpus: int = 0,
            eps_start: float = 1.0,
            eps_end: float = 0.02,
            eps_last_frame: int = 150000,
            sync_rate: int = 1000,
            gamma: float = 0.99,
            learning_rate: float = 1e-4,
            batch_size: int = 32,
            replay_size: int = 100000,
            warm_start_size: int = 10000,
            num_samples: int = 500,
            n_steps=4
    ):
        """
        PyTorch Lightning implementation of `N-Step DQN <http://incompleteideas.net/papers/sutton-88-with-erratum.pdf>`_

        Paper authors: Richard Sutton

        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:

            >>> from pl_bolts.models.reinforcement.n_step_dqn.model import NStepDQN
            ...
            >>> model = NStepDQN("PongNoFrameskip-v4")

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
            learning_rate: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            replay_size: total capacity of the replay buffer
            warm_start_size: how many random steps through the environment to be carried out at the start of
            training to fill the buffer with a starting point
            num_samples: the number of samples to pull from the dataset iterator and feed to the DataLoader
            n_steps: number of steps to approximate and use in the bellman update
        """
        super().__init__(env, gpus, eps_start, eps_end, eps_last_frame, sync_rate, gamma, learning_rate,
                         batch_size, replay_size, warm_start_size, num_samples)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.source = NStepExperienceSource(
            self.env, self.agent, device, n_steps=n_steps
        )
