# Double DQN

The original DQN tends to overestimate Q values during the Bellman update, leading to instability and is harmful to
training. This is due to the max operation in the Bellman equation. 

We are constantly taking the max of our agents estimates 
during our update. This may seem reasonable, if we could trust these estimates. However during the early stages of 
training, the estimates for these values will be off center and can lead to instability in training until
our estimates become more reliable

The Double DQN fixes this overestimation by choosing actions for the next state using the main trained network
but uses the values of these actions from the more stable target network. So we are still going to take the greedy
action, but the value will be less "optimisitc" because it is chosen by the target network.

## DQN expected return

````text
Q(s_t, a_t) = r_t + gamma * maxQ'(S_t+1, a)
````

## Double DQN expected return

````text
Q(s_t, a_t) = r_t + gamma * maxQ'(S_t+1, argmaxQ(S_t+1, a))
````

In order to update the original DQN to DoubleDQN we need to do the following 

### Update the Loss

````python
    def dqn_mse_loss(self, batch):

        states, actions, rewards, dones, next_states = batch  # batch of experiences, batch_size = 16

        actions_v = actions.unsqueeze(-1)  # adds a dimension, 16 -> [16, 1]
        output = self.net(states)  # shape [16, 2], [batch, action space]

        # gather the value of the outputs according to the actions index from the batch
        state_action_values = output.gather(1, actions_v).squeeze(-1)

        # dont want to mess with gradients when using the target network
        with torch.no_grad():
            next_outputs = self.net(next_states)  # [16, 2], [batch, action_space]

            next_state_acts = next_outputs.max(1)[1].unsqueeze(-1)   # take action at the index with the highest value
            next_tgt_out = self.target_net(next_states)

            # Take the value of the action chosen by the train network
            next_state_values = next_tgt_out.gather(1, next_state_acts).squeeze(-1)
            next_state_values[dones] = 0.0  # any steps flagged as done get a 0 value
            next_state_values = next_state_values.detach()  # remove values from the graph, no grads needed

        # calc expected discounted return of next_state_values
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        # Standard MSE loss between the state action values of the current state and the
        # expected state action values of the next state
        return nn.MSELoss()(state_action_values, expected_state_action_values)
````

## Results

### Double DQN Results
![Double DQN Results](../../docs/images/pong_double_dqn_baseline_results.png)

### DQN vs Double DQN

orange: DQN

blue: Double DQN

![Double DQN Results](../../docs/images/dqn_ddqn_comparison.png)

## References

[Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)