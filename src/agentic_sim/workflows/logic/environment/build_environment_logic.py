import random

from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.environment.environment import Environment
from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    EnvironmentState,
    InstitutionState,
    MarketState,
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
                market_id=f"market_{i:03d}",
                market_name=f"Market {i}",
                market_type="general",
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
                institution_id=f"institution_{i:03d}",
                institution_name=f"Institution {i}",
                market_id=market.market_id,
                available_liquidity=10_000.0,
                capital=5_000.0,
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
                    account_id=f"account_{account_number:05d}",
                    agent_id=agent.config.agent_id,
                    institution_id=institution.institution_id,
                    balance=1000.0,
                    outstanding_debt=0.0,
                    credit_limit=500.0,
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
