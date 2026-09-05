from dataclasses import dataclass, field

from agentic_sim.workflows.logic.environment.metadata.fields import MetadataField


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

    account_name: MetadataField[str]

    balance: MetadataField[float]
    outstanding_debt: MetadataField[float]
    credit_limit: MetadataField[float]


@dataclass
class CompanyState:
    """
    State of an economic company operating within a market.
    """

    company_id: int
    market_id: int

    company_name: MetadataField[str]

    capital: MetadataField[float]
    inventory: MetadataField[float]
    production_capacity: MetadataField[float]


@dataclass
class InstitutionState:
    """
    State of a financial institution.
    """

    institution_id: int
    market_id: int

    institution_name: MetadataField[str]

    available_liquidity: MetadataField[float]
    capital: MetadataField[float]


@dataclass
class MarketState:
    """
    State of an economic market visible to participating actors.
    """

    market_id: int

    market_name: MetadataField[str]

    price_level: MetadataField[float]
    available_supply: MetadataField[float]
