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
    learned latent opportunity representation.

    Input:

        [num_candidates, candidate_feature_dim]

    Output:

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


class ActionValueNetwork(nn.Module):
    """
    Predicts the expected long-term value of every currently
    available action candidate.

    The same current state embedding is paired with each action
    embedding.

    Conceptually:

        Q(s, a)

    Input:

        state_embedding:
            [state_embedding_dim]

        action_embeddings:
            [num_candidates, action_embedding_dim]

    Output:

        [num_candidates]

    The output layer is intentionally unbounded.
    """

    def __init__(
        self,
        *,
        state_embedding_dim: int,
        action_embedding_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.state_embedding_dim = state_embedding_dim

        self.action_embedding_dim = action_embedding_dim

        joint_dim = state_embedding_dim + action_embedding_dim

        self.network = nn.Sequential(
            nn.Linear(
                joint_dim,
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
                1,
            ),
        )

    def forward(
        self,
        state_embedding: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:

        if state_embedding.ndim != 1:
            raise ValueError(
                (
                    "State embedding must be one-dimensional "
                    "for single-agent opportunity evaluation | "
                    f"shape={tuple(state_embedding.shape)}"
                )
            )

        if action_embeddings.ndim != 2:
            raise ValueError(
                (
                    "Action embeddings must be two-dimensional | "
                    f"shape={tuple(action_embeddings.shape)}"
                )
            )

        if state_embedding.shape[0] != self.state_embedding_dim:
            raise ValueError(
                (
                    "Unexpected state embedding dimension | "
                    f"expected={self.state_embedding_dim} | "
                    f"actual={state_embedding.shape[0]}"
                )
            )

        if action_embeddings.shape[1] != self.action_embedding_dim:
            raise ValueError(
                (
                    "Unexpected action embedding dimension | "
                    f"expected={self.action_embedding_dim} | "
                    f"actual={action_embeddings.shape[1]}"
                )
            )

        num_candidates = action_embeddings.shape[0]

        if num_candidates == 0:
            return action_embeddings.new_empty((0,))

        state_batch = state_embedding.unsqueeze(0).expand(
            num_candidates,
            -1,
        )

        joint_representation = torch.cat(
            (
                state_batch,
                action_embeddings,
            ),
            dim=-1,
        )

        values = self.network(joint_representation)

        return values.squeeze(-1)


class ContrastiveLoss(nn.Module):
    """
    Siamese contrastive loss.

    label = 1:
        embeddings should be close.

    label = 0:
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

    This previously converted state embeddings directly into
    coordinates in the two-dimensional primitive action plane.

    It is no longer used by the candidate-based decision path.
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
    Neural components belonging to one economic agent.

    Active candidate-value architecture:

        observation features
                ↓
        StateEncoder
                ↓
        state_embedding
                │
                │
                ├────────────────────────┐
                │                        │
        candidate features               │
                ↓                        │
        ActionEncoder                    │
                ↓                        │
        action_embeddings                │
                │                        │
                └──────────────┬─────────┘
                               ↓
                     ActionValueNetwork
                               ↓
                 predicted candidate values
    """

    def __init__(
        self,
        embedding_dim: int = 3,
        encoder_hidden_dim: int = 512,
        action_embedding_dim: int = 16,
        action_encoder_hidden_dim: int = 128,
        value_hidden_dim: int = 128,
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
            embedding_dim=(action_embedding_dim),
            hidden_dim=(action_encoder_hidden_dim),
            dropout=dropout,
        )

        self.value_network = ActionValueNetwork(
            state_embedding_dim=(embedding_dim),
            action_embedding_dim=(action_embedding_dim),
            hidden_dim=(value_hidden_dim),
            dropout=dropout,
        )

        # Transitional legacy network.
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

    def encode_actions(
        self,
        action_features: torch.Tensor,
    ) -> torch.Tensor:

        return self.action_encoder(action_features)

    def score_actions(
        self,
        *,
        state_embedding: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce one predicted long-term value per candidate.
        """

        return self.value_network(
            state_embedding=state_embedding,
            action_embeddings=action_embeddings,
        )

    def evaluate_opportunities(
        self,
        *,
        observation_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Deterministic decision-time forward pass.

        Dropout is disabled during opportunity evaluation.

        Exploration should later be implemented explicitly
        during candidate selection rather than arising from
        stochastic neural-network layers.

        Returns:

            state_embedding
            action_embeddings
            predicted_values
        """

        was_training = self.training

        self.eval()

        try:

            with torch.no_grad():

                state_embedding = self.encode(observation_features)

                action_embeddings = self.encode_actions(action_features)

                predicted_values = self.score_actions(
                    state_embedding=(state_embedding),
                    action_embeddings=(action_embeddings),
                )

        finally:

            self.train(was_training)

        return (
            state_embedding,
            action_embeddings,
            predicted_values,
        )

    def encode_pair(
        self,
        observation_a: torch.Tensor,
        observation_b: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

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
        Legacy actor-based forward path.

        Retained temporarily until the candidate architecture
        completely replaces the old interface.
        """

        embedding = self.encoder(observation)

        return self.actor(embedding)
