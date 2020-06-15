# Noisy DQN

Up until now the DQN agent uses a seperate exploration policy, generally epsilon-greedy where start and end values
are set for its exploration. [Noisy Networks For Exploration](https://arxiv.org/abs/1706.10295) introduces
a new exploration strategy by adding noise parameters to the weightsof the fully connect layers which get updated 
during backpropagation of the network. The noise parameters drive 
the exploration of the network instead of simply taking random actions more frequently at the start of training and 
less frequently towards the end.The of authors of 
propose two ways of doing this.

During the optimization step a new set of noisy parameters are sampled. During training the agent acts according to
the fixed set of parameters. At the next optimization step, the parameters are updated with a new sample. This ensures
the agent always acts based on the parameters that are drawn from the current noise
distribution.

The authors propose two methods of injecting noise to the network.

1) Independent Gaussian Noise: This injects noise per weight. For each weight a random value is taken from the distribution. 
Noise parameters are stored inside the layer and are updated during backpropagation. The output of the layer is 
calculated as normal.

2) Factorized Gaussian Noise: This injects nosier per input/ouput. In order to minimize the number of random values 
this method stores two random vectors, one with the size of the input and the other with the size of the output. Using
these two vectors, a random matrix is generated for the layer by calculating the outer products of the vector


### Benefits

- Improved exploration function. Instead of just performing completely random actions, we add decreasing amount of noise
and uncertainty to our policy allowing to explore while still utilising its policy

- The fact that this method is automatically tuned means that we do not have to tune hyper parameters for
epsilon-greedy!

**Note:** for now I have just implemented the Independant Gaussian as it has been reported there isn't much difference
in results for these benchmark environments.

In order to update the basic DQN to a Noisy DQN we need to do the following

### Add Linear Layer

`````python
class NoisyLinear(nn.Linear):

    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        
        # init noise parameter for the weights with initial noise sigma
        weights = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(weights)
        epsilon_weight = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", epsilon_weight)

        if bias:
            # init noise parameter for the bias with initial noise sigma
            bias = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(bias)
            epsilon_bias = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", epsilon_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """initializes or resets the paramseter of the layer"""
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input_x: Tensor) -> Tensor:
        """
        Forward pass of the layer

        Args:
            input_x: input tensor

        Returns:
            output of the layer
        """
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            # add noise to layer bias
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        
        # add noise to layer weights
        noisy_weights = self.sigma_weight * self.epsilon_weight.data + self.weight

        return F.linear(input_x, noisy_weights, bias)
``````

### Update CNN with noisy layers
````python
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(
            NoisyLinear(conv_out_size, 512),    # use Noisy Linear Layer
            nn.ReLU(),
            NoisyLinear(512, n_actions)         # use Noisy Linear Layer
        )
````

## Results

The results below improved stability and faster performance growth. 

### Pong

#### Noisy DQN
 
Similar to the other improvements, the average score of the agent reaches positive numbers around the 250k mark and 
steadily increases till convergence. 
 
![Noisy DQN Results](../../docs/images/pong_noisy_dqn_results.png)

#### DQN vs Dueling DQN 

In comparison to the base DQN, the Noisy DQN is more stable and is able to converge on an optimal policy much faster
than the original. It seems that the replacement of the epsilon-greedy strategy with network noise provides a better
form of exploration.

 - Orange: DQN

 - Red: Noisy DQN

![Noisy DQN Comparison](../../docs/images/pong_noisy_dqn_comparison.png)

