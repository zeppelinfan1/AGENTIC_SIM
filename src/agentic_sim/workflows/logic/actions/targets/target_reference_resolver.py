from agentic_sim.workflows.logic.actions.targets.target_candidate import (
    TargetObjectType,
    TargetReference,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.environment.metadata.fields import (
    MetadataField,
)
from agentic_sim.workflows.logic.environment.perception.extractor import (
    VisibleTarget,
)


class TargetReferenceResolver:
    """
    Resolves stable TargetReference objects against the
    current Environment.

    Action components should use references rather than retaining
    mutable environment objects between stages.
    """

    TARGET_COLLECTIONS: dict[
        TargetObjectType,
        tuple[str, str],
    ] = {
        "account": (
            "accounts",
            "account_id",
        ),
        "institution": (
            "institutions",
            "institution_id",
        ),
        "market": (
            "markets",
            "market_id",
        ),
        "company": (
            "companies",
            "company_id",
        ),
    }

    def resolve_object(
        self,
        *,
        reference: TargetReference,
        environment: Environment,
    ) -> VisibleTarget:

        (
            collection_name,
            id_field,
        ) = self.TARGET_COLLECTIONS[reference.object_type]

        collection: list[VisibleTarget] = getattr(
            environment,
            collection_name,
        )

        matches = [
            target_object
            for target_object in collection
            if getattr(
                target_object,
                id_field,
            )
            == reference.object_id
        ]

        if not matches:
            raise ValueError(
                (
                    "Action target does not exist in current "
                    "environment | "
                    f"type={reference.object_type} | "
                    f"id={reference.object_id}"
                )
            )

        if len(matches) > 1:
            raise ValueError(
                (
                    "Action target ID is not unique | "
                    f"type={reference.object_type} | "
                    f"id={reference.object_id}"
                )
            )

        return matches[0]

    def resolve_metadata_field(
        self,
        *,
        reference: TargetReference,
        environment: Environment,
    ) -> MetadataField:

        target_object = self.resolve_object(
            reference=reference,
            environment=environment,
        )

        if not hasattr(
            target_object,
            reference.field_name,
        ):
            raise ValueError(
                (
                    "Action target field does not exist | "
                    f"type={reference.object_type} | "
                    f"id={reference.object_id} | "
                    f"field={reference.field_name}"
                )
            )

        field_value = getattr(
            target_object,
            reference.field_name,
        )

        if not isinstance(
            field_value,
            MetadataField,
        ):
            raise ValueError(
                (
                    "Action target field is not a MetadataField | "
                    f"type={reference.object_type} | "
                    f"id={reference.object_id} | "
                    f"field={reference.field_name}"
                )
            )

        return field_value
