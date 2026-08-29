from agentic_sim.workflows.task import Task
from agentic_sim.workflows.logic.agents.build_agents_logic import BuildAgents


class SimulationEngine(Task):

    def __init__(self, **kwargs) -> None:
        # Task initialization
        super().__init__(**kwargs)
        self.is_dev_run = self.get_bool_parameter("is_dev_run")
        self.dev_catalog = self.init_config["dev_catalog"]
        self.logger.info(
            "SimulationEngine parameters | is_dev_run=%s | dev_catalog=%s",
            self.is_dev_run,
            self.dev_catalog,
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
            init_config=self.init_config,
            spark=self.spark,
            logger=self.logger,
        )

    def run(self) -> None:
        self.logger.info("Starting simulation engine...")

        # Agent building logic
        self.build_agents.run()

        self.logger.info("Simulation engine completed.")
