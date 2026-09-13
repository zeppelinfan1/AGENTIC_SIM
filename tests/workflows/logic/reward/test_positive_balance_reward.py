import pytest

from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.environment.metadata.fields import (
    MetadataField,
)
from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    EnvironmentState,
)
from agentic_sim.workflows.logic.reward.models import (
    RewardContext,
)
from agentic_sim.workflows.logic.reward.positive_balance import (
    PositiveBalanceRewardMechanism,
)


def _account(
    *,
    account_id: int,
    actor_id: int,
    balance: float,
) -> AccountState:

    return AccountState(
        account_id=account_id,
        agent_id=actor_id,
        institution_id=1,
        account_name=MetadataField(
            value=f"Account {account_id}",
            visibility_type="private",
        ),
        balance=MetadataField(
            value=balance,
            visibility_type="private",
        ),
        outstanding_debt=MetadataField(
            value=0.0,
            visibility_type="private",
        ),
        credit_limit=MetadataField(
            value=500.0,
            visibility_type="private",
        ),
    )


def _environment(
    *,
    accounts: list[AccountState],
) -> Environment:

    return Environment(
        state=EnvironmentState(),
        markets=[],
        companies=[],
        institutions=[],
        accounts=accounts,
    )


def test_positive_balance_increase_produces_positive_reward():

    reward_mechanism = PositiveBalanceRewardMechanism()

    before_environment = _environment(
        accounts=[
            _account(
                account_id=1,
                actor_id=1,
                balance=1000.0,
            ),
        ],
    )

    after_environment = _environment(
        accounts=[
            _account(
                account_id=1,
                actor_id=1,
                balance=1100.0,
            ),
        ],
    )

    before = reward_mechanism.capture_state(
        actor_id=1,
        environment=before_environment,
    )

    after = reward_mechanism.capture_state(
        actor_id=1,
        environment=after_environment,
    )

    result = reward_mechanism.evaluate(
        context=RewardContext(
            actor_id=1,
            before=before,
            after=after,
        )
    )

    assert result.value == pytest.approx(100.0)


def test_positive_balance_decrease_produces_negative_reward():

    reward_mechanism = PositiveBalanceRewardMechanism()

    before = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=1000.0,
                ),
            ],
        ),
    )

    after = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=900.0,
                ),
            ],
        ),
    )

    result = reward_mechanism.evaluate(
        context=RewardContext(
            actor_id=1,
            before=before,
            after=after,
        )
    )

    assert result.value == pytest.approx(-100.0)


def test_reward_sums_all_agent_accounts():

    reward_mechanism = PositiveBalanceRewardMechanism()

    before = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=500.0,
                ),
                _account(
                    account_id=2,
                    actor_id=1,
                    balance=250.0,
                ),
                _account(
                    account_id=3,
                    actor_id=2,
                    balance=10000.0,
                ),
            ],
        ),
    )

    assert before.metrics["positive_balance"] == pytest.approx(750.0)


def test_negative_balances_do_not_count_toward_positive_balance():

    reward_mechanism = PositiveBalanceRewardMechanism()

    snapshot = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=500.0,
                ),
                _account(
                    account_id=2,
                    actor_id=1,
                    balance=-300.0,
                ),
            ],
        ),
    )

    assert snapshot.metrics["positive_balance"] == pytest.approx(500.0)


def test_other_agents_balances_do_not_affect_reward():

    reward_mechanism = PositiveBalanceRewardMechanism()

    before = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=1000.0,
                ),
                _account(
                    account_id=2,
                    actor_id=2,
                    balance=1000.0,
                ),
            ],
        ),
    )

    after = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=1000.0,
                ),
                _account(
                    account_id=2,
                    actor_id=2,
                    balance=5000.0,
                ),
            ],
        ),
    )

    result = reward_mechanism.evaluate(
        context=RewardContext(
            actor_id=1,
            before=before,
            after=after,
        )
    )

    assert result.value == pytest.approx(0.0)


def test_reward_scale_changes_magnitude_not_objective():

    reward_mechanism = PositiveBalanceRewardMechanism(
        reward_scale=0.01,
    )

    before = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=1000.0,
                ),
            ],
        ),
    )

    after = reward_mechanism.capture_state(
        actor_id=1,
        environment=_environment(
            accounts=[
                _account(
                    account_id=1,
                    actor_id=1,
                    balance=1100.0,
                ),
            ],
        ),
    )

    result = reward_mechanism.evaluate(
        context=RewardContext(
            actor_id=1,
            before=before,
            after=after,
        )
    )

    assert result.value == pytest.approx(1.0)

    assert result.components["positive_balance_change"] == pytest.approx(100.0)
