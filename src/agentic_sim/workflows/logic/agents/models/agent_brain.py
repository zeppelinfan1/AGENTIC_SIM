import torch
from torch import nn


class StateEncoder(nn.Module):
    """
    Encodes an agent observation into a learned latent
    decision-state representation.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.LazyLinear(
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(observation)


class ActionEncoder(nn.Module):
    """
    Encodes structured ActionCandidate features into a
    learned latent action/opportunity representation.

    The same ActionEncoder is applied to every currently
    available candidate for one agent.

    Input shape:

        [num_candidates, candidate_feature_dim]

    Output shape:

        [num_candidates, action_embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.LazyLinear(
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
        )

    def forward(
        self,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(action_features)


class ContrastiveLoss(nn.Module):
    """
    Siamese contrastive loss.

    label = 1
        embeddings should be close.

    label = 0
        embeddings should be separated by at least margin.
    """

    def __init__(
        self,
        margin: float = 1.0,
    ) -> None:
        super().__init__()

        self.margin = margin

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        distances = torch.linalg.vector_norm(
            embedding_a - embedding_b,
            dim=-1,
        )

        positive_loss = labels * distances.pow(2)

        negative_loss = (1 - labels) * torch.clamp(
            self.margin - distances,
            min=0.0,
        ).pow(2)

        return (positive_loss + negative_loss).mean()


class AgentActor(nn.Module):
    """
    Transitional legacy actor.

    Previously converted the state embedding directly into
    coordinates in the two-dimensional primitive action plane.

    SimulationEngine no longer uses this component for current
    candidate-based decision making.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        action_dim: int = 2,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                action_dim,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(embedding)


class AgentBrain(nn.Module):
    """
    Neural learning components belonging to one agent.

    Current active representation paths:

        observation
            ↓
        StateEncoder
            ↓
        state_embedding


        ActionCandidate features
            ↓
        ActionEncoder
            ↓
        action_embedding

    The next architecture stage will combine:

        state_embedding
        + action_embedding
            ↓
        ValueNetwork
            ↓
        predicted long-term candidate value
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        encoder_hidden_dim: int = 512,
        action_embedding_dim: int = 16,
        action_encoder_hidden_dim: int = 128,
        actor_hidden_dim: int = 128,
        action_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = StateEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=encoder_hidden_dim,
            dropout=dropout,
        )

        self.action_encoder = ActionEncoder(
            embedding_dim=action_embedding_dim,
            hidden_dim=action_encoder_hidden_dim,
            dropout=dropout,
        )

        # Transitional legacy component.
        self.actor = AgentActor(
            embedding_dim=embedding_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
        )

    def encode(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one perceived decision state.
        """

        return self.encoder(observation)

    def encode_actions(
        self,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode every candidate in a candidate-feature batch.
        """

        return self.action_encoder(action_features)

    def encode_pair(
        self,
        observation_a: torch.Tensor,
        observation_b: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Siamese state-encoding operation.

        Both observations pass through the same StateEncoder
        and therefore share weights.
        """

        embedding_a = self.encoder(observation_a)

        embedding_b = self.encoder(observation_b)

        return (
            embedding_a,
            embedding_b,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Legacy forward path.

        Retained temporarily until the candidate ValueNetwork
        fully replaces the old direct actor architecture.
        """

        embedding = self.encoder(observation)

        return self.actor(embedding)
