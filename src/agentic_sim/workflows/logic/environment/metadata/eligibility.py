from dataclasses import dataclass


@dataclass(frozen=True)
class IntentEligibilityMetadata:

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
