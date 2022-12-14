"""
PyTorch demo of PPO algorithm in discrete action space.
"""
import torch
import torch.nn as nn


class DiscretePolicyNetwork(nn.Module):
    def __init__(self, obs_shape: int, action_shape: int) -> None:
        """
        **Overview**:
            The definition of discrete action policy network used in PPO, which is mainly composed
            of three parts: encoder, mu and log_sigma.
        """
        # PyTorch necessary requirements for extending ``nn.Module`` .
        super(DiscretePolicyNetwork, self).__init__()
        # Define encoder module, which maps raw state into embedding vector.
        # It could be different for various state, such as Convolution Neural Network for image state.
        # Here we use one-layer MLP for vector state.
        self.encoder = nn.Sequential(
            nn.Linear(obs_shape, 32),
            nn.ReLU(),
        )
        # Define discrete action logit output network, just one-layer FC.
        self.head = nn.Linear(32, action_shape)

    # delimiter
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        **Overview**:
            The computation graph of discrete action policy network used in PPO.
        """
        # Transform original state into embedding vector (i.e. (B, *) -> (B, N)).
        x = self.encoder(x)
        # Calculate logit for each possible discrete action dimension.
        logit = self.head(x)
        return logit


# delimiter
def sample_action(logit: torch.Tensor) -> torch.Tensor:
    """
    **Overview**:
        The function of sampling discrete action, input shape = (B, action_shape), output shape = (B, ).
        In this example, batch_shape = (B, ), event_shape = (), sample_shape = ().
    """
    # Transform logit (raw output of policy network, e.g. last fully connected layer) into probability.
    prob = torch.softmax(logit, dim=-1)
    # Construct categorical distribution.
    # <link https://en.wikipedia.org/wiki/Categorical_distribution link>
    dist = torch.distributions.Categorical(probs=prob)
    # Sample one discrete action per sample and return it.
    return dist.sample()


# delimiter
def test_sample_action():
    """
    **Overview**:
        The function of testing sampling discrete action. Construct a naive policy and sample a group of action.
    """
    # Set batch_size = 4, obs_shape = 10, action_shape = 6.
    B, obs_shape, action_shape = 4, 10, 6
    # Generate state data from uniform distribution.
    state = torch.rand(B, obs_shape)
    # Define policy network with encoder and head.
    policy_network = DiscretePolicyNetwork(obs_shape, action_shape)
    # Policy network forward procedure, input state and output logit.
    logit = policy_network(state)
    assert logit.shape == (B, action_shape)
    # Sample action accoding to corresponding logit.
    action = sample_action(logit)
    assert action.shape == (B, )


if __name__ == "__main__":
    test_sample_action()
