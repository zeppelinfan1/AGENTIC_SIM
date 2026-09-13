import math

from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.reward.base import (
    RewardMechanism,
)
from agentic_sim.workflows.logic.reward.models import (
    RewardContext,
    RewardResult,
    RewardSnapshot,
)


class PositiveBalanceRewardMechanism(RewardMechanism):
    """
    V1 reward mechanism.

    Success is defined exclusively as increasing the total
    positive balance held across the agent's accounts.

    For each account:

        positive_balance = max(balance, 0)

    Agent-level balance metric:

        sum(positive_balance)

    Transition reward:

        positive_balance_after
        -
        positive_balance_before

    Examples:

        1000 -> 1100    reward = +100
        1000 -> 900     reward = -100
        1000 -> 1000    reward = 0

    This mechanism deliberately contains no truth orientation,
    fear avoidance, prediction error, or other psychological
    reward logic.

    It is a minimal baseline implementation.
    """

    METRIC_NAME = "positive_balance"

    def __init__(
        self,
        *,
        reward_scale: float = 1.0,
    ) -> None:

        if not math.isfinite(reward_scale):
            raise ValueError("reward_scale must be finite.")

        if reward_scale <= 0.0:
            raise ValueError(("reward_scale must be " "strictly positive."))

        self.reward_scale = float(reward_scale)

    def capture_state(
        self,
        *,
        actor_id: int,
        environment: Environment,
    ) -> RewardSnapshot:

        total_positive_balance = 0.0

        for account in environment.accounts:

            if account.agent_id != actor_id:
                continue

            balance = account.balance.value

            if isinstance(
                balance,
                bool,
            ):
                raise ValueError(
                    (
                        "Account balance must be numeric | "
                        f"account_id={account.account_id}"
                    )
                )

            if not isinstance(
                balance,
                (int, float),
            ):
                raise ValueError(
                    (
                        "Account balance must be numeric | "
                        f"account_id={account.account_id} | "
                        f"type={type(balance).__name__}"
                    )
                )

            numeric_balance = float(balance)

            if not math.isfinite(numeric_balance):
                raise ValueError(
                    (
                        "Account balance must be finite | "
                        f"account_id={account.account_id}"
                    )
                )

            total_positive_balance += max(
                numeric_balance,
                0.0,
            )

        return RewardSnapshot(
            actor_id=actor_id,
            metrics={
                self.METRIC_NAME: (total_positive_balance),
            },
        )

    def evaluate(
        self,
        *,
        context: RewardContext,
    ) -> RewardResult:

        if context.before.actor_id != context.actor_id:
            raise ValueError(
                (
                    "Before reward snapshot actor mismatch | "
                    f"context_actor={context.actor_id} | "
                    "snapshot_actor="
                    f"{context.before.actor_id}"
                )
            )

        if context.after.actor_id != context.actor_id:
            raise ValueError(
                (
                    "After reward snapshot actor mismatch | "
                    f"context_actor={context.actor_id} | "
                    "snapshot_actor="
                    f"{context.after.actor_id}"
                )
            )

        before_balance = float(context.before.metrics[self.METRIC_NAME])

        after_balance = float(context.after.metrics[self.METRIC_NAME])

        balance_change = after_balance - before_balance

        reward = balance_change * self.reward_scale

        return RewardResult(
            value=reward,
            components={
                "positive_balance_before": (before_balance),
                "positive_balance_after": (after_balance),
                "positive_balance_change": (balance_change),
                "reward_scale": (self.reward_scale),
            },
        )
