import random

from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.environment.environment import Environment
from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    EnvironmentState,
    InstitutionState,
    MarketState,
)
from agentic_sim.workflows.logic.environment.visibility.fields import (
    VisibilityField,
)


class BuildEnvironment:

    def __init__(
        self,
        num_markets: int,
        num_institutions: int,
        accounts_per_agent: int,
        random_seed: int,
        logger,
    ) -> None:

        self.num_markets = num_markets
        self.num_institutions = num_institutions
        self.accounts_per_agent = accounts_per_agent
        self.random_seed = random_seed
        self.logger = logger

        self.rng = random.Random(
            self.random_seed,
        )

    def _build_markets(
        self,
    ) -> list[MarketState]:

        return [
            MarketState(
                market_id=i,
                market_name=VisibilityField(
                    value=f"market_{i:03d}",
                    visibility_type="public",
                ),
                price_level=VisibilityField(
                    value=1.0,
                    visibility_type="public",
                ),
                available_supply=VisibilityField(
                    value=1000.0,
                    visibility_type="cascading",
                ),
            )
            for i in range(
                1,
                self.num_markets + 1,
            )
        ]

    def _build_institutions(
        self,
        markets: list[MarketState],
    ) -> list[InstitutionState]:

        institutions = []

        for i in range(
            1,
            self.num_institutions + 1,
        ):

            market = self.rng.choice(
                markets,
            )

            institution = InstitutionState(
                institution_id=i,
                market_id=market.market_id,
                institution_name=VisibilityField(
                    value=f"institution_{i:03d}",
                    visibility_type="public",
                ),
                available_liquidity=VisibilityField(
                    value=10_000.0,
                    visibility_type="restricted",
                ),
                capital=VisibilityField(
                    value=5_000.0,
                    visibility_type="restricted",
                ),
            )

            institutions.append(
                institution,
            )

        return institutions

    def _build_accounts(
        self,
        agents: list[Agent],
        institutions: list[InstitutionState],
    ) -> list[AccountState]:

        accounts = []

        account_number = 1

        for agent in agents:

            for _ in range(
                self.accounts_per_agent,
            ):

                institution = self.rng.choice(
                    institutions,
                )

                account = AccountState(
                    account_id=account_number,
                    agent_id=agent.config.agent_id,
                    institution_id=institution.institution_id,
                    account_name=VisibilityField(
                        value=f"Account {account_number}",
                        visibility_type="private",
                    ),
                    balance=VisibilityField(
                        value=1000.0,
                        visibility_type="private",
                    ),
                    outstanding_debt=VisibilityField(
                        value=0.0,
                        visibility_type="private",
                    ),
                    credit_limit=VisibilityField(
                        value=500.0,
                        visibility_type="private",
                    ),
                )

                accounts.append(
                    account,
                )

                account_number += 1

        return accounts

    def _validate_environment(
        self,
        agents: list[Agent],
        environment: Environment,
    ) -> None:

        agent_ids = {agent.config.agent_id for agent in agents}

        institution_ids = {
            institution.institution_id for institution in environment.institutions
        }

        market_ids = {market.market_id for market in environment.markets}

        for account in environment.accounts:

            if account.agent_id not in agent_ids:
                raise ValueError(f"Unknown agent: {account.agent_id}")

            if account.institution_id not in institution_ids:
                raise ValueError(f"Unknown institution: {account.institution_id}")

        for institution in environment.institutions:

            if institution.market_id not in market_ids:
                raise ValueError(f"Unknown market: {institution.market_id}")

    def run(
        self,
        agents: list[Agent],
    ) -> Environment:

        markets = self._build_markets()

        institutions = self._build_institutions(
            markets=markets,
        )

        accounts = self._build_accounts(
            agents=agents,
            institutions=institutions,
        )

        environment = Environment(
            state=EnvironmentState(),
            markets=markets,
            institutions=institutions,
            accounts=accounts,
        )

        self._validate_environment(
            environment=environment,
            agents=agents,
        )

        return environment
