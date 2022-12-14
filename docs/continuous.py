"""
PyTorch demo of PPO algorithm in continuous action space.
"""
from typing import Dict
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent


class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            The definition of continuous action policy network used in PPO, which is mainly composed
            of three parts: encoder, mu and log_sigma.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` .
        super(ContinuousPolicyNetwork, self).__init__()
        # Define encoder module, which maps raw state into embedding vector.
        # It could be different for various state, such as Convolution Neural Network for image state.
        # Here we use two-layer MLP for vector state.
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        # Define mu module, which is a FC and outputs the argument mu for gaussian distribution.
        self.mu = nn.Linear(32, action_shape)
        # Define log_sigma module, which is a learnable parameter but independent to state.
        self.log_sigma = nn.Parameter(torch.zeros(1, action_shape))

    # delimiter
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        **Overview**:
            The computation graph of continuous action policy network used in PPO.
        """
        # Transform original state into embedding vector (i.e. (B, *) -> (B, N)).
        x = self.encoder(x)
        # Output the argument mu depending on the embedding vector.
        mu = self.mu(x)
        # Utilize broadcast mechanism to make the same shape between log_sigma and mu.
        # ``zeros_like`` operation doesn't pass gradient.
        # <link https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html#in-brief-tensor-broadcasting link>
        log_sigma = self.log_sigma + torch.zeros_like(mu)
        # Utilize exponential operation to produce the actual sigma.
        sigma = torch.exp(log_sigma)
        return {'mu': mu, 'sigma': sigma}


# delimiter
def sample_continuous_action(logit: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    **Overview**:
        The function of sampling continuous action, input is a dict with two keys 'mu' and 'sigma',
        both of them has shape = (B, action_shape), output shape = (B, action_shape).
        In this example, batch_shape = (B, ), event_shape = (action_shape, ), sample_shape = ().
    """
    # Construct gaussian distribution with $$\mu, \sigma$$
    # <link https://en.wikipedia.org/wiki/Normal_distribution link>
    dist = Normal(logit['mu'], logit['sigma'])
    # Reinterpret ``action_shape`` gaussian distribution into a multivariate gaussian distribution with
    # diagonal convariance matrix.
    # Ensure each event is independent with each other.
    # <link https://pytorch.org/docs/stable/distributions.html#independent link>
    dist = Independent(dist, 1)
    # Sample one action of the shape ``action_shape`` per sample and return it.
    return dist.sample()


# delimiter
def test_sample_continuous_action():
    """
    **Overview**:
        The function of testing sampling continuous action. Construct a standard continuous action
        policy and sample a group of action.
    """
    # Set batch_size = 4, obs_shape = 10, action_shape = 6.
    # ``action_shape`` is different from discrete and continuous action. The former is the possible
    # choice of a discrete action while the latter is the dimension of continuous action.
    B, obs_shape, action_shape = 4, 10, 6
    # Generate state data from uniform distribution.
    state = torch.rand(B, obs_shape)
    # Define continuous action network (which is similar to reparameterization) with encoder, mu and log_sigma.
    policy_network = ContinuousPolicyNetwork(obs_shape, action_shape)
    # Policy network forward procedure, input state and output dict-type logit.
    logit = policy_network(state)
    assert isinstance(logit, dict)
    assert logit['mu'].shape == (B, action_shape)
    assert logit['sigma'].shape == (B, action_shape)
    # Sample action accoding to corresponding logit (i.e., mu and sigma).
    action = sample_continuous_action(logit)
    assert action.shape == (B, action_shape)


if __name__ == "__main__":
    test_sample_continuous_action()
