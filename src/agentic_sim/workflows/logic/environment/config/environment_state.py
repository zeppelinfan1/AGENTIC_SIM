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
