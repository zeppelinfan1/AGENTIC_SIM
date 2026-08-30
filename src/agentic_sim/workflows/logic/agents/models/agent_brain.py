import torch
from torch import nn


class StateEncoder(nn.Module):
    """
    Encodes an agent observation into a latent representation.

    Conceptually mirrors the old NumPy network:
        input -> dense -> ReLU -> dense -> ReLU -> dropout -> embedding
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class ContrastiveLoss(nn.Module):
    """
    Siamese contrastive loss.

    label = 1 -> embeddings should be close
    label = 0 -> embeddings should be separated by at least `margin`
    """

    def __init__(self, margin: float = 1.0) -> None:
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
            dim=1,
        )

        positive_loss = labels * distances.pow(2)

        negative_loss = (1 - labels) * torch.clamp(
            self.margin - distances,
            min=0.0,
        ).pow(2)

        return (positive_loss + negative_loss).mean()


class AgentActor(nn.Module):
    """
    Converts the learned state representation into coordinates
    in the primitive action space.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 128,
        action_dim: int = 2,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding)


class AgentBrain(nn.Module):
    """
    Complete decision network for an agent.

    observation
        ↓
    StateEncoder
        ↓
    latent representation
        ↓
    AgentActor
        ↓
    primitive action coordinates
    """

    def __init__(
        self,
        observation_dim: int,
        embedding_dim: int = 3,
        encoder_hidden_dim: int = 512,
        actor_hidden_dim: int = 128,
        action_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = StateEncoder(
            input_dim=observation_dim,
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
        return self.encoder(observation)

    def encode_pair(
        self,
        observation_a: torch.Tensor,
        observation_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Siamese operation.

        Both observations pass through the SAME encoder,
        therefore both branches share weights.
        """
        embedding_a = self.encoder(observation_a)
        embedding_b = self.encoder(observation_b)

        return embedding_a, embedding_b

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.encoder(observation)
        action = self.actor(embedding)

        return action
