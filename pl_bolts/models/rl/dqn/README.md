# Deep Q Network 

The DQN was introduced in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) by 
researchers at DeepMind. This took the concept of tabular Q learning and scaled it to much larger problems by 
apporximating the Q function using a deep neural network. 

The goal behind DQN was to take the simple control method of Q learning and scale it up in order to solve complicated
tasks. As well as this, the method needed to be stable. The DQN solves these issues with the following additions.

### 1. Approximated Q Function
Storing Q values in a table works well in theory, but is completely unscalable. Instead, the authors apporximate the 
Q function using a deep neural network. This allows the DQN to be used for much more complicated tasks

### 2. Replay Buffer
Similar to supervised learning, the DQN learns on randomly sampled batches of previous data stored in an 
Experience Replay Buffer. The 'target' is calculated using the Bellman equation 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Q(s,a)<-(r+{\gamma}\max_{a'{\in}A}Q(s',a'))^2"/>

and then we optimize using SGD just
like a standard supervised learning problem.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;L=(Q(s,a)-(r+{\gamma}\max_{a'{\in}A}Q(s',a'))^2"/>

### 

## Benefits

## Implementation

## Results
![DQN Basline Results](../../docs/images/pong_dqn_baseline_results.png)

## References
 


