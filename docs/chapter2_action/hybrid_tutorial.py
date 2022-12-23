"""
PyTorch demo of PPO algorithm in hybrid action space.
"""
from typing import Dict
import torch
import torch.nn as nn
import treetensor.torch as ttorch
from torch.distributions import Normal, Independent


class HybridPolicyNetwork(nn.Module):
    def __init__(self, obs_shape: int, action_shape: Dict[str, int]) -> None:
        """
        **Overview**:
            The definition of hybrid action policy network used in PPO, which is mainly composed
            of three parts: encoder, action_type head (discrete) and action_args head (continuous).
        """
        # PyTorch necessary requirements for extending ``nn.Module`` .
        super(HybridPolicyNetwork, self).__init__()
        # Define encoder module, which maps raw state into embedding vector.
        # It could be different for various state, such as Convolution Neural Network for image state.
        # Here we use two-layer MLP for vector state.
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        # Define action_type head module, which outputs discrete logit.
        self.action_type_shape = action_shape['action_type_shape']
        self.action_type_head = nn.Linear(32, self.action_type_shape)
        # Define action_args head module, which outputs corresponding continuous action arguments.
        self.action_args_shape = action_shape['action_args_shape']
        self.action_args_mu = nn.Linear(32, self.action_args_shape)
        self.action_args_log_sigma = nn.Parameter(torch.zeros(1, self.action_args_shape))

    # delimiter
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        **Overview**:
            The computation graph of hybrid action policy network used in PPO.
        """
        # Transform original state into embedding vector (i.e. (B, *) -> (B, N)).
        x = self.encoder(x)
        # Output discrete action logit.
        logit = self.action_type_head(x)
        # Output the argument mu depending on the embedding vector.
        mu = self.action_args_mu(x)
        # Utilize broadcast mechanism to make the same shape between log_sigma and mu.
        # ``zeros_like`` operation doesn't pass gradient.
        # <link https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html#in-brief-tensor-broadcasting link>
        log_sigma = self.action_args_log_sigma + torch.zeros_like(mu)
        # Utilize exponential operation to produce the actual sigma.
        sigma = torch.exp(log_sigma)
        # Return treetensor-type output.
        return ttorch.as_tensor({
            'action_type': logit,
            'action_args': {
                'mu': mu,
                'sigma': sigma
            }
        })


# delimiter
def sample_hybrid_action(logit: ttorch.Tensor) -> torch.Tensor:
    """
    **Overview**:
        The function of sampling hybrid action, input is a dict with two keys 'mu' and 'sigma',
        both of them has shape = (B, action_shape), output shape = (B, action_shape).
        In this example, batch_shape = (B, ), event_shape = (action_shape, ), sample_shape = ().
    """
    # Transform logit (raw output of discrete policy head, e.g. last fully connected layer) into probability.
    prob = torch.softmax(logit.action_type, dim=-1)
    # Construct categorical distribution.
    # <link https://en.wikipedia.org/wiki/Categorical_distribution link>
    discrete_dist = torch.distributions.Categorical(probs=prob)
    # Sample one discrete action type per sample
    action_type = discrete_dist.sample()

    # Construct gaussian distribution with $$\mu, \sigma$$
    # <link https://en.wikipedia.org/wiki/Normal_distribution link>
    continuous_dist = Normal(logit.action_args.mu, logit.action_args.sigma)
    # Reinterpret ``action_shape`` gaussian distribution into a multivariate gaussian distribution with
    # diagonal convariance matrix.
    # Ensure each event is independent with each other.
    # <link https://pytorch.org/docs/stable/distributions.html#independent link>
    continuous_dist = Independent(continuous_dist, 1)
    # Sample one action args of the shape ``action_shape`` per sample.
    action_args = continuous_dist.sample()
    # Return the final parameterized action.
    return ttorch.as_tensor({
        'action_type': action_type,
        'action_args': action_args,
    })


# delimiter
def test_sample_hybrid_action():
    """
    **Overview**:
        The function of testing sampling hybrid action. Construct a hybrid action (parameterized action)
        policy and sample a group of action.
    """
    # Set batch_size = 4, obs_shape = 10, action_shape is a dict, including 3 possible discrete
    # action types and 3 corresponding continuous arguments. The relationship between action_type
    # and action_args are represented by the below ``mask`` .
    B, obs_shape, action_shape = 4, 10, {'action_type_shape': 3, 'action_args_shape': 3}
    mask = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
    # Generate state data from uniform distribution.
    state = torch.rand(B, obs_shape)
    # Define hybrid action network with encoder, discrete head and continuous head.
    policy_network = HybridPolicyNetwork(obs_shape, action_shape)
    # Policy network forward procedure, input state and output dict-type logit.
    logit = policy_network(state)
    assert isinstance(logit, ttorch.Tensor)
    assert logit.action_type.shape == (B, action_shape['action_type_shape'])
    assert logit.action_args.mu.shape == (B, action_shape['action_args_shape'])
    assert logit.action_args.sigma.shape == (B, action_shape['action_args_shape'])
    # Sample action accoding to corresponding logit part.
    action = sample_hybrid_action(logit)
    assert action.action_type.shape == (B, )
    assert action.action_args.shape == (B, action_shape['action_args_shape'])
    # Acquire each sample's mask by looking up in ``mask`` with action type。
    data_mask = torch.as_tensor([mask[i] for i in action.action_type]).bool()
    # Filter corresponding action_args according to mask and re-assign it.
    filtered_action_args = ttorch.masked_select(action.action_args, data_mask)
    action.action_args = filtered_action_args
    assert action.action_args.shape == (B, )
    # Select some samples (for example)
    selected_action = action[1:3]
    assert selected_action.action_type.shape == (2, )


if __name__ == "__main__":
    test_sample_hybrid_action()
