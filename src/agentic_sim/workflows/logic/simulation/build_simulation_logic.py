from dataclasses import dataclass

from agentic_sim.workflows.logic.agents.agent import Agent
from agentic_sim.workflows.logic.agents.build_agents_logic import (
    BuildAgents,
)
from agentic_sim.workflows.logic.environment.build_environment_logic import (
    BuildEnvironment,
)
from agentic_sim.workflows.logic.environment.environment import (
    Environment,
)
from agentic_sim.workflows.logic.environment.perception.extractor import (
    VisibleFieldExtractor,
)
from agentic_sim.workflows.logic.environment.perception.perception import (
    Perception,
)


@dataclass
class SimulationContext:
    """
    Runtime components required by the simulation loop.
    """

    agents: list[Agent]
    environment: Environment
    perception: Perception


class BuildSimulation:
    """
    Builds the initial simulation runtime.

    Responsible for:
        - agent population
        - environment
        - stable perception topology

    Does not execute simulation steps.
    """

    def __init__(
        self,
        *,
        num_agents: int,
        num_markets: int,
        num_companies: int,
        num_institutions: int,
        accounts_per_agent: int,
        random_seed: int,
        spark,
        logger,
    ) -> None:

        self.logger = logger

        self.build_agents = BuildAgents(
            num_agents=num_agents,
            spark=spark,
            logger=logger,
        )

        self.build_environment = BuildEnvironment(
            num_markets=num_markets,
            num_institutions=num_institutions,
            num_companies=num_companies,
            accounts_per_agent=accounts_per_agent,
            random_seed=random_seed,
            logger=logger,
        )

        self.perception = Perception(
            visible_field_extractor=VisibleFieldExtractor(),
        )

    def run(self) -> SimulationContext:

        self.logger.info("Starting simulation build...")

        # 1. Build agents
        agents = self.build_agents.run()

        self.logger.info(
            "Agent population initialized | count=%s",
            len(agents),
        )

        # 2. Build shared environment
        environment = self.build_environment.run(
            agents=agents,
        )

        self.logger.info(
            (
                "Environment initialized | "
                "markets=%s | "
                "companies=%s | "
                "institutions=%s | "
                "accounts=%s"
            ),
            len(environment.markets),
            len(environment.companies),
            len(environment.institutions),
            len(environment.accounts),
        )

        # 3. Build stable perception topology
        self.perception.build(
            agents=agents,
            environment=environment,
        )

        self.logger.info(
            "Perception topology initialized | agents=%s",
            len(self.perception.linkages),
        )

        self.logger.info("Simulation build completed.")

        return SimulationContext(
            agents=agents,
            environment=environment,
            perception=self.perception,
        )
