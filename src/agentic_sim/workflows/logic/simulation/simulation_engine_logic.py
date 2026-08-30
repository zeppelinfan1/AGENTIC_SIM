from agentic_sim.workflows.task import Task
from agentic_sim.workflows.logic.agents.build_agents_logic import BuildAgents


class SimulationEngine(Task):

    NAMED_PARAMETER_KEYS = (
        "is_dev_run",
        "dev_catalog",
        "num_agents",
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.is_dev_run = self.get_bool_parameter("is_dev_run")
        self.dev_catalog = self.init_config["dev_catalog"]
        self.num_agents = int(self.init_config["num_agents"])
        self.logger.info(
            "SimulationEngine parameters | is_dev_run=%s | dev_catalog=%s | num_agents=%s",
            self.is_dev_run,
            self.dev_catalog,
            self.num_agents,
        )

        # Validate current spark environment
        current_environment = self.spark.sql("""
            SELECT
                current_catalog() AS current_catalog,
                current_schema() AS current_schema
            """).first()
        if current_environment is None:
            raise RuntimeError("Spark environment query returned no rows.")
        self.logger.info(
            "Spark connection successful | catalog=%s | schema=%s",
            current_environment["current_catalog"],
            current_environment["current_schema"],
        )

        # Class initialization
        self.build_agents = BuildAgents(
            spark=self.spark,
            logger=self.logger,
            num_agents=self.num_agents,
        )

    def run(self) -> None:
        self.logger.info("Starting simulation engine...")

        self.agents = self.build_agents.run()

        self.logger.info(
            "Agent population initialized | num_agents=%s",
            len(self.agents),
        )

        self.logger.info("Simulation engine completed.")
