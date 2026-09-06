from dataclasses import dataclass


@dataclass(frozen=True)
class IntentEligibilityMetadata:
    """
    Field-level physical eligibility for primitive action
    directions.

    The historical class name contains "Intent", but these
    flags should now be understood as environmental action
    physics rather than agent strategy or intention.

    Eligibility answers:

        "Can this primitive transformation meaningfully
        participate in an action involving this field?"

    It does not determine desirability, magnitude, selection,
    or expected value.
    """

    create: bool = False
    destroy: bool = False
    produce: bool = False
    consume: bool = False

    @property
    def has_any_eligibility(
        self,
    ) -> bool:
        return any(
            (
                self.create,
                self.destroy,
                self.produce,
                self.consume,
            )
        )
