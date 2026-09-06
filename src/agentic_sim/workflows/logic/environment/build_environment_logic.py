import random

from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.environment.environment import Environment
from agentic_sim.workflows.logic.environment.metadata.eligibility import (
    IntentEligibilityMetadata,
)
from agentic_sim.workflows.logic.environment.metadata.fields import (
    MetadataField,
)
from agentic_sim.workflows.logic.environment.state import (
    AccountState,
    CompanyState,
    EnvironmentState,
    InstitutionState,
    MarketState,
)


class BuildEnvironment:

    def __init__(
        self,
        num_markets: int,
        num_institutions: int,
        num_companies: int,
        accounts_per_agent: int,
        random_seed: int,
        logger,
    ) -> None:

        self.num_markets = num_markets
        self.num_institutions = num_institutions
        self.num_companies = num_companies
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
                market_name=MetadataField(
                    value=f"market_{i:03d}",
                    visibility_type="public",
                ),
                price_level=MetadataField(
                    value=1.0,
                    visibility_type="public",
                ),
                available_supply=MetadataField(
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
                institution_name=MetadataField(
                    value=f"institution_{i:03d}",
                    visibility_type="public",
                ),
                available_liquidity=MetadataField(
                    value=10_000.0,
                    visibility_type="restricted",
                ),
                capital=MetadataField(
                    value=5_000.0,
                    visibility_type="restricted",
                ),
            )

            institutions.append(
                institution,
            )

        return institutions

    def _build_companies(
        self,
        markets: list[MarketState],
    ) -> list[CompanyState]:

        companies = []

        for i in range(
            1,
            self.num_companies + 1,
        ):

            market = self.rng.choice(
                markets,
            )

            company = CompanyState(
                company_id=i,
                market_id=market.market_id,
                company_name=MetadataField(
                    value=f"company_{i:03d}",
                    visibility_type="public",
                ),
                capital=MetadataField(
                    value=10_000.0,
                    visibility_type="restricted",
                ),
                inventory=MetadataField(
                    value=500.0,
                    visibility_type="cascading",
                    eligibility=IntentEligibilityMetadata(
                        produce=True,
                        consume=True,
                    ),
                ),
                production_capacity=MetadataField(
                    value=100.0,
                    visibility_type="cascading",
                    eligibility=IntentEligibilityMetadata(
                        create=True,
                        destroy=True,
                    ),
                ),
            )

            companies.append(
                company,
            )

        return companies

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
                    account_name=MetadataField(
                        value=f"Account {account_number}",
                        visibility_type="private",
                    ),
                    balance=MetadataField(
                        value=1000.0,
                        visibility_type="private",
                        eligibility=IntentEligibilityMetadata(
                            consume=True,
                        ),
                    ),
                    outstanding_debt=MetadataField(
                        value=0.0,
                        visibility_type="private",
                    ),
                    credit_limit=MetadataField(
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
                raise ValueError(("Unknown institution: " f"{account.institution_id}"))

        for institution in environment.institutions:

            if institution.market_id not in market_ids:
                raise ValueError(("Unknown market: " f"{institution.market_id}"))

        for company in environment.companies:

            if company.market_id not in market_ids:
                raise ValueError(("Unknown market: " f"{company.market_id}"))

    def run(
        self,
        agents: list[Agent],
    ) -> Environment:

        markets = self._build_markets()

        companies = self._build_companies(
            markets=markets,
        )

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
            companies=companies,
            institutions=institutions,
            accounts=accounts,
        )

        self._validate_environment(
            environment=environment,
            agents=agents,
        )

        return environment
