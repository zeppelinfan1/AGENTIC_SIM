import torch
from torch import nn


class StateEncoder(nn.Module):
    """
    Encodes an agent observation into a latent representation.

    Current structure:

        input
        -> dense
        -> ReLU
        -> dense
        -> ReLU
        -> dropout
        -> embedding
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
        return self.network(
            observation,
        )


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

        positive_loss = labels * distances.pow(
            2,
        )

        negative_loss = (1 - labels) * torch.clamp(
            self.margin - distances,
            min=0.0,
        ).pow(
            2,
        )

        return (positive_loss + negative_loss).mean()


class AgentActor(nn.Module):
    """
    Transitional legacy actor.

    This network previously converted the latent state directly
    into coordinates in the primitive action plane.

    The current simulation loop no longer uses this actor because
    decisions are moving toward explicit dynamic ActionCandidate
    evaluation.

    It remains temporarily so the old interface does not need to
    be deleted before the ActionEncoder / ValueNetwork replacement
    exists.
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
        return self.network(
            embedding,
        )


class AgentBrain(nn.Module):
    """
    Agent learning components.

    Current active decision-state path:

        observation
            ↓
        StateEncoder
            ↓
        state embedding

    The simulation now builds ActionCandidate objects separately.

    The next architecture stage will add:

        ActionCandidate
            ↓
        ActionEncoder
            ↓
        candidate representation

        state representation
        + candidate representation
            ↓
        ValueNetwork
            ↓
        predicted long-term value

    AgentActor remains temporarily for backwards compatibility
    but is no longer called by SimulationEngine.
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        encoder_hidden_dim: int = 512,
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

        self.actor = AgentActor(
            embedding_dim=embedding_dim,
            hidden_dim=actor_hidden_dim,
            action_dim=action_dim,
        )

    def encode(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            observation,
        )

    def encode_pair(
        self,
        observation_a: torch.Tensor,
        observation_b: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Siamese operation.

        Both observations pass through the same encoder,
        therefore both branches share weights.
        """

        embedding_a = self.encoder(
            observation_a,
        )

        embedding_b = self.encoder(
            observation_b,
        )

        return (
            embedding_a,
            embedding_b,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Legacy forward path retained temporarily.

        SimulationEngine no longer uses this method for
        candidate-based decision making.
        """

        embedding = self.encoder(
            observation,
        )

        return self.actor(
            embedding,
        )
