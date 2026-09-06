from agentic_sim.workflows.logic.actions.magnitude.magnitude_constraint_resolver import (
    MagnitudeConstraintResolver,
)
from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetReference,
)
from agentic_sim.workflows.logic.actions.targets.target_reference_resolver import (
    TargetReferenceResolver,
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
from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    EnvironmentState,
)


def _build_environment(
    *,
    balance: float,
    eligibility: IntentEligibilityMetadata,
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
            eligibility=eligibility,
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


def _resolver() -> MagnitudeConstraintResolver:
    return MagnitudeConstraintResolver(
        target_reference_resolver=(TargetReferenceResolver()),
    )


def _reference() -> TargetReference:
    return TargetReference(
        object_type="account",
        object_id=1,
        field_name="balance",
    )


def test_consume_is_capped_at_current_positive_quantity():
    environment = _build_environment(
        balance=100.0,
        eligibility=IntentEligibilityMetadata(
            consume=True,
        ),
    )

    constraint = _resolver().resolve(
        actor_id=1,
        target=_reference(),
        direction=PrimitiveDirection.CONSUME,
        environment=environment,
    )

    assert constraint is not None
    assert constraint.minimum == 0.0
    assert constraint.maximum == 100.0


def test_destroy_or_consume_cannot_remove_from_zero_quantity():
    environment = _build_environment(
        balance=0.0,
        eligibility=IntentEligibilityMetadata(
            consume=True,
        ),
    )

    constraint = _resolver().resolve(
        actor_id=1,
        target=_reference(),
        direction=PrimitiveDirection.CONSUME,
        environment=environment,
    )

    assert constraint is None


def test_create_uses_positive_floor_when_current_quantity_is_zero():
    environment = _build_environment(
        balance=0.0,
        eligibility=IntentEligibilityMetadata(
            create=True,
        ),
    )

    constraint = _resolver().resolve(
        actor_id=1,
        target=_reference(),
        direction=PrimitiveDirection.CREATE,
        environment=environment,
    )

    assert constraint is not None
    assert constraint.minimum == 0.0
    assert constraint.maximum == 1.0


def test_ineligible_direction_produces_no_constraint():
    environment = _build_environment(
        balance=100.0,
        eligibility=IntentEligibilityMetadata(
            consume=True,
        ),
    )

    constraint = _resolver().resolve(
        actor_id=1,
        target=_reference(),
        direction=PrimitiveDirection.PRODUCE,
        environment=environment,
    )

    assert constraint is None
