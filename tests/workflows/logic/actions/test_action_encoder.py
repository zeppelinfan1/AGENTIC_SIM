import torch

from agentic_sim.workflows.logic.agents.models.agent_brain import (
    ActionEncoder,
)


def test_action_encoder_encodes_every_candidate():

    encoder = ActionEncoder(
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    candidate_features = torch.randn(
        7,
        24,
    )

    embeddings = encoder(candidate_features)

    assert embeddings.shape == (
        7,
        16,
    )


def test_action_encoder_preserves_empty_batch_shape():

    encoder = ActionEncoder(
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )

    candidate_features = torch.empty(
        0,
        24,
    )

    embeddings = encoder(candidate_features)

    assert embeddings.shape == (
        0,
        16,
    )
