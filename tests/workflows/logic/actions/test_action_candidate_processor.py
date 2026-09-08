import math

import pytest

from agentic_sim.workflows.logic.actions.action_candidate import (
    ActionCandidate,
)
from agentic_sim.workflows.logic.actions.encoding.action_candidate_processor import (
    ActionCandidateProcessor,
)
from agentic_sim.workflows.logic.actions.primitives import (
    PrimitiveDirection,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetReference,
)
from agentic_sim.workflows.logic.agents.config.agent_config import (
    AgentObservation,
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
                    "visible_fields": {
                        "account_name": "Account 1",
                        "balance": 100.0,
                        "outstanding_debt": 0.0,
                        "credit_limit": 500.0,
                    },
                }
            ],
            "institutions": [],
            "markets": [],
            "companies": [],
        },
    )


def _candidate(
    *,
    magnitude: float = 25.0,
) -> ActionCandidate:

    return ActionCandidate(
        target=TargetReference(
            object_type="account",
            object_id=1,
            field_name="balance",
        ),
        direction=(PrimitiveDirection.CONSUME),
        magnitude=magnitude,
    )


def test_processor_returns_one_row_per_candidate():

    processor = ActionCandidateProcessor()

    candidates = [
        _candidate(magnitude=25.0),
        _candidate(magnitude=50.0),
        _candidate(magnitude=75.0),
    ]

    processed = processor.process_many(
        candidates=candidates,
        observation=_observation(),
    )

    assert processed.values.shape == (
        3,
        len(processed.feature_names),
    )

    assert processed.candidates == tuple(candidates)


def test_processor_encodes_object_type_direction_and_field():

    processor = ActionCandidateProcessor()

    processed = processor.process_many(
        candidates=[_candidate()],
        observation=_observation(),
    )

    row = processed.values[0]

    feature_values = dict(
        zip(
            processed.feature_names,
            row.tolist(),
            strict=True,
        )
    )

    assert feature_values["target_object_type.account"] == 1.0

    assert feature_values["target_object_type.company"] == 0.0

    assert feature_values["direction.consume"] == 1.0

    assert feature_values["direction.create"] == 0.0

    assert feature_values["target_field.account.balance"] == 1.0


def test_processor_encodes_scaled_numeric_context():

    processor = ActionCandidateProcessor()

    processed = processor.process_many(
        candidates=[_candidate(magnitude=25.0)],
        observation=_observation(),
    )

    feature_values = dict(
        zip(
            processed.feature_names,
            processed.values[0].tolist(),
            strict=True,
        )
    )

    assert feature_values["target_value_signed_log1p"] == pytest.approx(
        math.log1p(100.0)
    )

    assert feature_values["magnitude_log1p"] == pytest.approx(math.log1p(25.0))

    assert feature_values["magnitude_relative_to_target_scale"] == pytest.approx(0.25)

    assert feature_values["target_value_is_zero"] == 0.0


def test_raw_object_id_is_not_neural_feature():

    processor = ActionCandidateProcessor()

    assert not any(
        "object_id" in feature_name for feature_name in processor.feature_names
    )


def test_processor_rejects_target_field_not_visible_to_agent():

    processor = ActionCandidateProcessor()

    candidate = ActionCandidate(
        target=TargetReference(
            object_type="account",
            object_id=1,
            field_name="balance",
        ),
        direction=(PrimitiveDirection.CONSUME),
        magnitude=25.0,
    )

    observation = AgentObservation(
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

    with pytest.raises(ValueError):
        processor.process_many(
            candidates=[candidate],
            observation=observation,
        )


def test_processor_handles_empty_candidate_set():

    processor = ActionCandidateProcessor()

    processed = processor.process_many(
        candidates=[],
        observation=_observation(),
    )

    assert processed.values.shape == (
        0,
        len(processor.feature_names),
    )

    assert processed.candidates == ()
