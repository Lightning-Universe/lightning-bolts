# N Step DQN

N Step DQN was introduced in [Learning to Predict by the Methods
of Temporal Differences 
](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf). This method improves upon the original DQN by updating 
our Q values with the expected reward from multiple steps in the
future as opposed to the expected reward from the immediate next state. When getting the Q values for a state action 
pair using a single step which looks like this

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s_t,a_t)=r_t+{\gamma}\max_aQ(s_t+1,a_t+1)"/>

but because the Q function is recursive we can continue to roll this out into multiple steps, looking at the expected
return for each step into the future. 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s_t,a_t)=r_t+{\gamma}r_{t+1}+{\gamma}^2\max_{a'}Q(s_{t+2},a')"/>

The above example shows a 2-Step look ahead, but this could be rolled out to the end of the episode, which is just 
Monte Carlo learning. Although we could just do a monte carlo update and look forward to the end of the episode, it 
wouldn't be a good idea. Every time we take another step into the future, we are basing our approximation off our 
current policy. For a large portion of training, our policy is going to be less than optimal. For example, at the start
of training, our policy will be in a state of high exploration, and will be little better than random. 

---
**NOTE**

For each rollout step you must scale the discount factor accordingly by the number of steps. As you can see from the 
equation above, the second gamma value is to the power of 2. If we rolled this out one step further, we would use 
gamma to the power of 3 and so.

---

So if we are aproximating future rewards off a bad policy, chances are those approximations are going to be pretty 
bad and every time we unroll our update equation, the worse it will get. The fact that we are using an off policy method
like DQN with a large replay buffer will make this even worse, as there is a high chance that we will be training on 
experiences using an old policy that was worse than our current policy.

So we need to strike a balance between looking far enough ahead to improve the convergence of our agent, but not so far 
that are updates become unstable. In general, small values of 2-4 work best.  

### Benefits

- Multi-Step learning is capable of learning faster than typical 1 step learning methods.
- Note that this method introduces a new hyperparameter n. Although n=4 is generally a good starting point and provides
good results across the board.

### Implementation

#### Multi Step Buffer

The only thing we need to change for the N Step DQN is the buffer. We need a multi step 
buffer that combines n-steps into a single experience. This requires the following

##### N Step Buffer:

Unlike the standard buffer, we need to use 2 buffers. One to store the n step roll outs
and another to hold the accumulated multi step experience. 

##### Append:
The append function needs to be changed. If the n_step_buffer is not full, i.e we dont have
enough experiences to make a multi step experience, then we just append our current experience
to the n_step_buffer.

If the n_step_buffer has enough experiences, then we can take the last n steps and form an 
accumulate multi step experience to be added to the buffer.

The multi step experience will look like the following:

- State = the state at the start of the n_step buffer
- Action = the action at the start of the n_step_buffer
- Reward = is the accumulated discounted reward over the last n steps
- Next State = the next state at the end of the n_step_buffer
- Done = the done flag at the end of the n_step_buffer

````python

    def append(self, experience) -> None:
        """
        add an experience to the buffer by collecting n steps of experiences
        Args:
            experience: tuple (state, action, reward, done, next_state)
        """
        self.n_step_buffer.append(experience)

        if len(self.n_step_buffer) >= self.n_step:
            reward, next_state, done = self.get_transition_info()
            first_experience = self.n_step_buffer[0]
            multi_step_experience = Experience(first_experience.state,
                                               first_experience.action,
                                               reward,
                                               done,
                                               next_state)

            self.buffer.append(multi_step_experience)

    def get_transition_info(self, gamma=0.9) -> Tuple[np.float, np.array, np.int]:
        """
        get the accumulated transition info for the n_step_buffer
        Args:
            gamma: discount factor

        Returns:
            multi step reward, final observation and done
        """
        last_experience = self.n_step_buffer[-1]
        final_state = last_experience.new_state
        done = last_experience.done
        reward = last_experience.reward

        # calculate reward
        # in reverse order, go through all the experiences up till the first experience
        for experience in reversed(list(self.n_step_buffer)[:-1]):
            reward_t = experience.reward
            new_state_t = experience.new_state
            done_t = experience.done

            reward = reward_t + gamma * reward * (1 - done_t)
            final_state, done = (new_state_t, done_t) if done_t else (final_state, done)

        return reward, final_state, done
````

##### Sample

The sample function will behave as normal

A snippet of the code for the N Step Replay Buffer is shown below

## Results

As excpected, the N-Step DQN converges much faster than the standard DQN, however it also adds more instability to the 
loss of the agent. This can be seen in the following experiments. 

### Pong

#### N-Step DQN

The N-Step DQN shows the greates increase in performance with respect to the other DQN variations. After less than 150k steps the agent begins to
consistently win games and achieves the top score after ~170K steps. This is reflected in the sharp peak of the 
total episode steps and of course, the total episode rewards.
 
![N-Step DQN Baseline Results](../../docs/images/pong_nstep_dqn_1.png)

#### DQN vs N-Step DQN 

This improvement is shown in stark contrast to the base DQN, which only begins to win games after 250k steps and 
requires over twice as many steps (450k) as the N-Step agent to achieve the high score of 21. One important thing to 
notice is the large increase in the loss of the N-Step agent. This is expected as the agent is building 
its expected reward off approximations of the future states. The large the size of N, the greater the instability. 
Previous literature, listed below, shows the best results for the Pong environment with an N step between 3-5. For these
experiments I opted with an N step of 4.

![N-Step DQN Baseline Results vs DQN Baseline Results](../../docs/images/pong_nstep_dqn_2.png)


## References
 - [Learning to Predict by the Methods of Temporal Differences ](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
 - [Deep Reinforcement Learning Hands On: Second Edition - Chapter 08 ](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition)
 - [Rainbow Is All You Need](https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/07.n_step_learning.ipynb)