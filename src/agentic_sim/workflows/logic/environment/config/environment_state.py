from dataclasses import dataclass, field


@dataclass
class EnvironmentState:
    """
    State of the environment.
    """

    step: int = 0


@dataclass
class MarketState:
    """
    State of the market.
    """

    market_name: str
    price_level: float = 1.0
    available_supply: float = 1000.0


@dataclass
class InstitutionState:
    """
    State of an institution.
    """

    institution_name: str
    cash: float
    assets: float
    debt: float


@dataclass
class AccountState:
    """
    State of an account held by an agent at an institution.
    """

    account_id: str
    owner_agent_name: str
    institution_name: str

    balance: float = 0.0
    outstanding_debt: float = 0.0
    credit_limit: float = 0.0
