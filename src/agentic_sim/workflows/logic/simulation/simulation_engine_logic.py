from agentic_sim.workflows.task import Task

from agentic_sim.workflows.logic.agents.build_agents_logic import (
    BuildAgents,
)
from agentic_sim.workflows.logic.environment.build_environment_logic import (
    BuildEnvironment,
)
from agentic_sim.workflows.logic.environment.extractor import (
    VisibleFieldExtractor,
)
from agentic_sim.workflows.logic.environment.perception import (
    Perception,
)


class SimulationEngine(Task):

    NAMED_PARAMETER_KEYS: tuple[str, ...] = (
        "is_dev_run",
        "dev_catalog",
        "num_agents",
        "num_markets",
        "num_institutions",
        "accounts_per_agent",
        "random_seed",
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Runtime parameters
        self.is_dev_run = self.get_bool_parameter(
            "is_dev_run",
        )
        self.dev_catalog = self.init_config["dev_catalog"]
        self.num_agents = int(self.init_config["num_agents"])
        self.num_markets = int(self.init_config["num_markets"])
        self.num_institutions = int(self.init_config["num_institutions"])
        self.accounts_per_agent = int(self.init_config["accounts_per_agent"])
        self.random_seed = int(self.init_config["random_seed"])
        self.logger.info(
            (
                "SimulationEngine parameters | "
                "is_dev_run=%s | "
                "dev_catalog=%s | "
                "num_agents=%s | "
                "num_markets=%s | "
                "num_institutions=%s | "
                "accounts_per_agent=%s | "
                "random_seed=%s"
            ),
            self.is_dev_run,
            self.dev_catalog,
            self.num_agents,
            self.num_markets,
            self.num_institutions,
            self.accounts_per_agent,
            self.random_seed,
        )

        # Validate Spark environment
        current_environment = self.spark.sql("""
            SELECT
                current_catalog() AS current_catalog,
                current_schema() AS current_schema
        """).first()

        if current_environment is None:
            raise RuntimeError("Spark environment query returned no rows.")

        self.logger.info(
            ("Spark connection successful | " "catalog=%s | schema=%s"),
            current_environment["current_catalog"],
            current_environment["current_schema"],
        )

        # Component initialization
        self.build_agents = BuildAgents(
            num_agents=self.num_agents,
            spark=self.spark,
            logger=self.logger,
        )

        self.build_environment = BuildEnvironment(
            num_markets=self.num_markets,
            num_institutions=self.num_institutions,
            accounts_per_agent=self.accounts_per_agent,
            random_seed=self.random_seed,
            logger=self.logger,
        )

        self.visible_field_extractor = VisibleFieldExtractor()

        self.perception = Perception(
            visible_field_extractor=self.visible_field_extractor,
        )

    def run(self) -> None:
        self.logger.info("Starting simulation engine...")

        # 1. Build agents
        agents = self.build_agents.run()
        self.logger.info(
            "Agent population initialized | count=%s",
            len(agents),
        )

        # 2. Build environment
        environment = self.build_environment.run(
            agents=agents,
        )
        self.logger.info(
            (
                "Environment initialized | "
                "markets=%s | "
                "institutions=%s | "
                "accounts=%s"
            ),
            len(environment.markets),
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

        # 4. Generate current perceptions
        observations = []

        for agent in agents:

            observation = self.perception.perceive(
                agent=agent,
                environment=environment,
            )
            observations.append(
                observation,
            )
            self.logger.info(
                (
                    "Agent perception generated | "
                    "agent_id=%s | "
                    "observation_type=%s"
                ),
                agent.config.agent_id,
                observation.observation_type,
            )

        self.logger.info(
            ("Simulation initialization complete | " "agents=%s | " "observations=%s"),
            len(agents),
            len(observations),
        )
        self.logger.info("Simulation engine completed.")
