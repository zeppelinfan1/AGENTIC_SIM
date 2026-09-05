from dataclasses import dataclass


@dataclass(frozen=True)
class IntentEligibilityMetadata:
    """
    Declares which intent directions may meaningfully
    operate on or involve a state field.

    This describes possibility, not strategy.
    """

    create: bool = False
    destroy: bool = False
    produce: bool = False
    consume: bool = False
