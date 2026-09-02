from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

T = TypeVar("T")


VisibilityType = Literal[
    "public",
    "private",
    "cascading",
    "restricted",
]


@dataclass(frozen=True)
class VisibilityField(Generic[T]):
    """
    Environment value with its associated visibility rule.
    """

    value: T
    visibility_type: VisibilityType
