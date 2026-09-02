from dataclasses import dataclass, field

from agentic_sim.workflows.logic.environment.visibility.fields import VisibilityField


@dataclass
class EnvironmentState:
    """
    State of the environment.
    """

    step: int = 0


@dataclass
class AccountState:
    """
    State of an account held by an agent at an institution.
    """

    account_id: int
    agent_id: int
    institution_id: int

    account_name: VisibilityField[str]

    balance: VisibilityField[float]
    outstanding_debt: VisibilityField[float]
    credit_limit: VisibilityField[float]


@dataclass
class InstitutionState:
    """
    State of a financial institution.
    """

    institution_id: int
    market_id: int

    institution_name: VisibilityField[str]

    available_liquidity: VisibilityField[float]
    capital: VisibilityField[float]


@dataclass
class MarketState:
    """
    State of an economic market visible to participating actors.
    """

    market_id: int

    market_name: VisibilityField[str]

    price_level: VisibilityField[float]
    available_supply: VisibilityField[float]
