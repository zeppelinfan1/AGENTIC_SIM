from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_sim.workflows.logic.environment.visibility.resolver import (
        VisibilityResolver,
    )


class VisibilityRule(ABC):
    """
    Base visibility strategy.
    """

    @abstractmethod
    def is_visible(
        self,
        *,
        actor_id: int,
        target_object: object,
        resolver: "VisibilityResolver",
    ) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class PublicVisibility(VisibilityRule):
    """
    Field is visible to every actor.
    """

    def is_visible(
        self,
        *,
        actor_id: int,
        target_object: object,
        resolver: "VisibilityResolver",
    ) -> bool:
        return True


@dataclass(frozen=True)
class RestrictedVisibility(VisibilityRule):
    """
    Field is not visible to external actors.
    """

    def is_visible(
        self,
        *,
        actor_id: int,
        target_object: object,
        resolver: "VisibilityResolver",
    ) -> bool:
        return False


@dataclass(frozen=True)
class PrivateVisibility(VisibilityRule):
    """
    Field is visible only when the actor has a direct
    relationship with the target object.
    """

    def is_visible(
        self,
        *,
        actor_id: int,
        target_object: object,
        resolver: "VisibilityResolver",
    ) -> bool:
        return resolver.is_directly_linked(
            actor_id=actor_id,
            target_object=target_object,
        )


@dataclass(frozen=True)
class CascadingVisibility(VisibilityRule):
    """
    Field is visible when the actor can reach the target
    through the permitted environment relationship chain.
    """

    def is_visible(
        self,
        *,
        actor_id: int,
        target_object: object,
        resolver: "VisibilityResolver",
    ) -> bool:
        return resolver.is_cascading_linked(
            actor_id=actor_id,
            target_object=target_object,
        )
