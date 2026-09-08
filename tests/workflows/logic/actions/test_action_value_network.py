import pytest
import torch

from agentic_sim.workflows.logic.agents.models.agent_brain import (
    ActionValueNetwork,
)


def test_value_network_returns_one_value_per_candidate():

    network = ActionValueNetwork(
        state_embedding_dim=3,
        action_embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    state_embedding = torch.randn(3)

    action_embeddings = torch.randn(
        7,
        16,
    )

    values = network(
        state_embedding=state_embedding,
        action_embeddings=action_embeddings,
    )

    assert values.shape == (7,)


def test_value_network_supports_single_candidate():

    network = ActionValueNetwork(
        state_embedding_dim=3,
        action_embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    values = network(
        state_embedding=torch.randn(3),
        action_embeddings=torch.randn(
            1,
            16,
        ),
    )

    assert values.shape == (1,)


def test_value_network_supports_empty_candidate_set():

    network = ActionValueNetwork(
        state_embedding_dim=3,
        action_embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    values = network(
        state_embedding=torch.randn(3),
        action_embeddings=torch.empty(
            0,
            16,
        ),
    )

    assert values.shape == (0,)


def test_value_network_is_differentiable():

    network = ActionValueNetwork(
        state_embedding_dim=3,
        action_embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    state_embedding = torch.randn(
        3,
        requires_grad=True,
    )

    action_embeddings = torch.randn(
        4,
        16,
        requires_grad=True,
    )

    values = network(
        state_embedding=state_embedding,
        action_embeddings=action_embeddings,
    )

    loss = values.sum()

    loss.backward()

    assert state_embedding.grad is not None
    assert action_embeddings.grad is not None

    assert any(parameter.grad is not None for parameter in network.parameters())


def test_value_network_rejects_wrong_state_dimension():

    network = ActionValueNetwork(
        state_embedding_dim=3,
        action_embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    with pytest.raises(ValueError):
        network(
            state_embedding=torch.randn(4),
            action_embeddings=torch.randn(
                5,
                16,
            ),
        )


def test_value_network_rejects_wrong_action_dimension():

    network = ActionValueNetwork(
        state_embedding_dim=3,
        action_embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    with pytest.raises(ValueError):
        network(
            state_embedding=torch.randn(3),
            action_embeddings=torch.randn(
                5,
                15,
            ),
        )
