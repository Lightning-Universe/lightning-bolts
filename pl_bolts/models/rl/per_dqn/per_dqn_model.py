"""
# Prioritized Experience Replay

The standard DQN uses a buffer to break up the correlation between experiences and uniform random samples for each
batch. Instead of just randomly sampling from the buffer prioritized experience replay (PER) prioritizes these samples
based on training loss. This concept was introduced in the paper
[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

Essentially we want to train more on the samples that suprise the agent.

The priority of each sample is defined below where


.. math::

    P(i)=\frac{P^\alpha_i}{\sum_kP_k^\alpha}

where pi is the priority of the ith sample in the buffer and
𝛼 is the number that shows how much emphasis we give to the priority. If 𝛼 = 0 , our
sampling will become uniform as in the classic DQN method. Larger values for 𝛼 put
more stress on samples with higher priority

Its important that new samples are set to the highest priority so that they are sampled soon. This however introduces
bias to new samples in our dataset. In order to compensate for this bias, the value of the weight is defined as

.. math::

    w_i=(N . P(i))^{-\beta}

Wher beta is a hyper parameter between 0-1. When beta is 1 the bias is fully compensated. However authors noted that
in practice it is better to start beta with a small value near 0 and slowly increase it to 1.

### Benefits

- The benefits of this technique are that the agent sees more samples that it struggled with and gets more
chances to improve upon it.

### PER Memory Buffer

First step is to replace the standard experience replay buffer with the prioritized experience replay buffer. This
is pretty large (100+ lines) so I wont go through it here. There are two buffers implemented. The first is a naive
list based buffer found in memory.PERBuffer and the second is more efficient buffer using a Sum Tree datastructure.

The list based version is simpler, but has a sample complexity of O(N). The Sum Tree in comparison has a complexity
of O(1) for sampling and O(logN) for updating priorities.

### Update loss function

The next thing we do is to use the sample weights that we get from PER. Add the following code to the end of the
loss function. This applies the weights of our sample to the batch loss. Then we return the mean loss and weighted loss
for each datum, with the addition of a small epsilon value.

````python

    # explicit MSE loss
    loss = (state_action_values - expected_state_action_values) ** 2

    # weighted MSE loss
    weighted_loss = batch_weights * loss

    # return the weighted_loss for the batch and the updated weighted loss for each datum in the batch
    return weighted_loss.mean(), (weighted_loss + 1e-5).data.cpu().numpy()

````

## Results

The results below show improved stability and faster performance growth.

### Pong

#### PER DQN

Similar to the other improvements, we see that PER improves the stability of the agents training and shows to converged
on an optimal policy faster.

![Noisy DQN Results](../../docs/images/pong_per_dqn_baseline_v1_results.png)

#### DQN vs PER DQN

In comparison to the base DQN, the PER DQN does show improved stability and performance. As expected, the loss
of the PER DQN is siginificantly lower. This is the main objective of PER by focusing on experiences with high loss.

It is important to note that loss is not the only metric we should be looking at. Although the agent may have very
low loss during training, it may still perform poorly due to lack of exploration.


![Noisy DQN Comparison](../../docs/images/pong_per_dqn_baseline_v1_results_comp.png)

 - Orange: DQN

 - Pink: PER DQN

"""

from collections import OrderedDict
from typing import Tuple, List

import torch

from pl_bolts.models.rl.common.experience import ExperienceSource, PrioRLDataset
from pl_bolts.models.rl.common.memory import PERBuffer
from pl_bolts.models.rl.dqn.dqn_model import DQN


class PERDQN(DQN):
    """
    PyTorch Lightning implementation of `DQN With Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_

    Paper authors: Tom Schaul, John Quan, Ioannis Antonoglou, David Silver

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:

            >>> from pl_bolts.models.rl.per_dqn.per_dqn_model import PERDQN
            ...
            >>> model = PERDQN("PongNoFrameskip-v4")

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

        .. note::
            This example is based on: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter08/05_dqn_prio_replay.py

        .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

        """

    def training_step(self, batch, _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        samples, indices, weights = batch

        indices = indices.cpu().numpy()

        self.agent.update_epsilon(self.global_step)

        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss, batch_weights = self.loss(samples, weights)

        # update priorities in buffer
        self.buffer.update_priorities(indices, batch_weights)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(self.device),
            "avg_reward": torch.tensor(self.avg_reward),
            "train_loss": loss,
            "episode_steps": torch.tensor(self.total_episode_steps),
        }
        status = {
            "steps": torch.tensor(self.global_step).to(self.device),
            "avg_reward": torch.tensor(self.avg_reward),
            "total_reward": torch.tensor(self.total_reward).to(self.device),
            "episodes": self.episode_count,
            "episode_steps": self.episode_steps,
            "epsilon": self.agent.epsilon,
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_weights: List
    ) -> Tuple[torch.Tensor, List]:
        """
        Calculates the mse loss with the priority weights of the batch from the PER buffer

        Args:
            batch: current mini batch of replay data
            batch_weights: how each of these samples are weighted in terms of priority

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        batch_weights = torch.tensor(batch_weights)

        actions_v = actions.unsqueeze(-1)
        state_action_vals = self.net(states).gather(1, actions_v)
        state_action_vals = state_action_vals.squeeze(-1)
        with torch.no_grad():
            next_s_vals = self.target_net(next_states).max(1)[0]
            next_s_vals[dones] = 0.0
            exp_sa_vals = next_s_vals.detach() * self.gamma + rewards
        loss = (state_action_vals - exp_sa_vals) ** 2
        losses_v = batch_weights * loss
        return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()

    def prepare_data(self) -> None:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        device = torch.device(self.trainer.root_gpu) if self.trainer.gpus >= 1 else self.device
        self.source = ExperienceSource(self.env, self.agent, device)
        self.buffer = PERBuffer(self.replay_size)
        self.populate(self.warm_start_size)

        self.dataset = PrioRLDataset(self.buffer, self.batch_size)
