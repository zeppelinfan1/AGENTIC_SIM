from agentic_sim.workflows.logic.actions.candidates.action_candidate_provider import (
    ActionCandidateProvider,
)
from agentic_sim.workflows.logic.actions.magnitude.magnitude_candidate_generator import (
    MagnitudeCandidateGenerator,
)
from agentic_sim.workflows.logic.actions.magnitude.magnitude_constraint_resolver import (
    MagnitudeConstraintResolver,
)
from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate_provider import (
    TargetCandidateProvider,
)
from agentic_sim.workflows.logic.actions.targets.target_reference_resolver import (
    TargetReferenceResolver,
)
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.environment.metadata.eligibility import (
    IntentEligibilityMetadata,
)
from agentic_sim.workflows.logic.environment.metadata.fields import (
    MetadataField,
)
from agentic_sim.workflows.logic.environment.perception.extractor import (
    VisibleFieldExtractor,
)
from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    EnvironmentState,
)


def _build_environment(
    *,
    balance: float = 100.0,
) -> Environment:

    account = AccountState(
        account_id=1,
        agent_id=1,
        institution_id=1,
        account_name=MetadataField(
            value="Account 1",
            visibility_type="private",
        ),
        balance=MetadataField(
            value=balance,
            visibility_type="private",
            eligibility=IntentEligibilityMetadata(
                create=True,
                consume=True,
            ),
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

    return Environment(
        state=EnvironmentState(),
        markets=[],
        companies=[],
        institutions=[],
        accounts=[
            account,
        ],
    )


def _observation() -> AgentObservation:
    return AgentObservation(
        agent_name="Agent 1",
        observation_type="transactional_agent",
        observation_parameters={
            "agent_id": 1,
            "environment_step": 0,
            "accounts": [
                {
                    "account_id": 1,
                    "institution_id": 1,
                    "visible_fields": {},
                }
            ],
            "institutions": [],
            "markets": [],
            "companies": [],
        },
    )


def _providers():
    target_reference_resolver = TargetReferenceResolver()

    target_provider = TargetCandidateProvider(
        visible_field_extractor=(VisibleFieldExtractor()),
    )

    action_provider = ActionCandidateProvider(
        magnitude_constraint_resolver=(
            MagnitudeConstraintResolver(
                target_reference_resolver=(target_reference_resolver),
            )
        ),
        magnitude_candidate_generator=(MagnitudeCandidateGenerator()),
    )

    return (
        target_provider,
        action_provider,
    )


def test_action_candidates_are_generated_only_for_eligible_directions():
    environment = _build_environment(
        balance=100.0,
    )

    target_provider, action_provider = _providers()

    target_candidates = target_provider.get_candidates(
        observation=_observation(),
        environment=environment,
    )

    action_candidates = action_provider.get_candidates(
        actor_id=1,
        target_candidates=target_candidates,
        environment=environment,
    )

    assert (
        len(
            target_candidates,
        )
        == 1
    )

    assert (
        len(
            action_candidates,
        )
        == 8
    )

    create_candidates = [
        candidate
        for candidate in action_candidates
        if candidate.direction == PrimitiveDirection.CREATE
    ]

    consume_candidates = [
        candidate
        for candidate in action_candidates
        if candidate.direction == PrimitiveDirection.CONSUME
    ]

    assert [candidate.magnitude for candidate in create_candidates] == [
        25.0,
        50.0,
        75.0,
        100.0,
    ]

    assert [candidate.magnitude for candidate in consume_candidates] == [
        25.0,
        50.0,
        75.0,
        100.0,
    ]

    assert all(
        candidate.direction
        not in {
            PrimitiveDirection.DESTROY,
            PrimitiveDirection.PRODUCE,
        }
        for candidate in action_candidates
    )


def test_candidate_generation_is_deterministic():
    environment = _build_environment(
        balance=100.0,
    )

    target_provider, action_provider = _providers()

    target_candidates = target_provider.get_candidates(
        observation=_observation(),
        environment=environment,
    )

    first = action_provider.get_candidates(
        actor_id=1,
        target_candidates=target_candidates,
        environment=environment,
    )

    second = action_provider.get_candidates(
        actor_id=1,
        target_candidates=target_candidates,
        environment=environment,
    )

    assert first == second


def test_magnitude_resolution_uses_current_environment_state():
    environment = _build_environment(
        balance=100.0,
    )

    target_provider, action_provider = _providers()

    target_candidates = target_provider.get_candidates(
        observation=_observation(),
        environment=environment,
    )

    # TargetCandidate was discovered while balance=100.
    assert target_candidates[0].current_value == 100.0

    # Change current mutable environment state after discovery.
    environment.accounts[0].balance = MetadataField(
        value=40.0,
        visibility_type="private",
        eligibility=IntentEligibilityMetadata(
            create=True,
            consume=True,
        ),
    )

    action_candidates = action_provider.get_candidates(
        actor_id=1,
        target_candidates=target_candidates,
        environment=environment,
    )

    create_magnitudes = [
        candidate.magnitude
        for candidate in action_candidates
        if candidate.direction == PrimitiveDirection.CREATE
    ]

    consume_magnitudes = [
        candidate.magnitude
        for candidate in action_candidates
        if candidate.direction == PrimitiveDirection.CONSUME
    ]

    assert create_magnitudes == [
        10.0,
        20.0,
        30.0,
        40.0,
    ]

    assert consume_magnitudes == [
        10.0,
        20.0,
        30.0,
        40.0,
    ]
