"""
PyTorch implementation of "sampling discrete action" in PPO
"""
import torch
import torch.nn as nn


def sample_action(logits: torch.Tensor) -> torch.Tensor:
    """
    **Overview**:
        The function of sampling discrete action, input shape = (B, action_shape), output shape = (B, ).
        In this example, batch_shape = (B, ), event_shape = (), sample_shape = ().
    """
    # Transform logit (raw output of policy network, e.g. last fully connected layer) into probability.
    probs = torch.softmax(logits, dim=-1)
    # Construct categorical distribution.
    # link: https://en.wikipedia.org/wiki/Categorical_distribution
    dist = torch.distributions.Categorical(probs=probs)
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
    # Define policy network, we use a two-layer MLP for example.
    policy_network = nn.Sequential(
        nn.Linear(10, 16),
        nn.ReLU(),
        nn.Linear(16, action_shape)
    )
    # Policy network forward procedure, input state and output logits.
    logits = policy_network(state)
    assert logits.shape == (B, action_shape)
    # Sample action accoding to corresponding logits.
    action = sample_action(logits)
    assert action.shape == (B, )


if __name__ == "__main__":
    test_sample_action()
