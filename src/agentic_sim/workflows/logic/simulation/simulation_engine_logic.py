from agentic_sim.workflows.task import Task

from agentic_sim.workflows.logic.simulation.build_simulation_logic import (
    BuildSimulation,
    SimulationContext,
)


class SimulationEngine(Task):

    NAMED_PARAMETER_KEYS: tuple[str, ...] = (
        "is_dev_run",
        "dev_catalog",
        "num_steps",
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

        self.num_steps = int(
            self.init_config["num_steps"],
        )

        self.num_agents = int(
            self.init_config["num_agents"],
        )

        self.num_markets = int(
            self.init_config["num_markets"],
        )

        self.num_institutions = int(
            self.init_config["num_institutions"],
        )

        self.accounts_per_agent = int(
            self.init_config["accounts_per_agent"],
        )

        self.random_seed = int(
            self.init_config["random_seed"],
        )

        self.logger.info(
            (
                "SimulationEngine parameters | "
                "is_dev_run=%s | "
                "dev_catalog=%s | "
                "num_steps=%s | "
                "num_agents=%s | "
                "num_markets=%s | "
                "num_institutions=%s | "
                "accounts_per_agent=%s | "
                "random_seed=%s"
            ),
            self.is_dev_run,
            self.dev_catalog,
            self.num_steps,
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

        # Simulation builder
        self.build_simulation = BuildSimulation(
            num_agents=self.num_agents,
            num_markets=self.num_markets,
            num_institutions=self.num_institutions,
            accounts_per_agent=self.accounts_per_agent,
            random_seed=self.random_seed,
            spark=self.spark,
            logger=self.logger,
        )

    def _run_step(
        self,
        *,
        simulation: SimulationContext,
    ) -> None:

        step = simulation.environment.state.step

        self.logger.info(
            "Starting simulation step | step=%s",
            step,
        )

        for agent in simulation.agents:

            # 1. Perceive current environment
            observation = simulation.perception.perceive(
                agent=agent,
                environment=simulation.environment,
            )

            self.logger.info(
                (
                    "Agent perception generated | "
                    "step=%s | "
                    "agent_id=%s | "
                    "observation_type=%s"
                ),
                step,
                agent.config.agent_id,
                observation.observation_type,
            )

            # Future:
            #
            # 2. Convert observation to tensor
            # 3. StateEncoder creates embedding
            # 4. AgentActor chooses action
            # 5. Submit action to environment

        # Future:
        #
        # 6. Environment resolves actions
        # 7. Calculate rewards
        # 8. Perform learning

        self.logger.info(
            "Simulation step completed | step=%s",
            step,
        )

    def run(self) -> None:

        self.logger.info("Starting simulation engine...")

        simulation = self.build_simulation.run()

        for _ in range(self.num_steps):

            self._run_step(
                simulation=simulation,
            )

            simulation.environment.state.step += 1

        self.logger.info(
            ("Simulation engine completed | " "steps_completed=%s | " "final_step=%s"),
            self.num_steps,
            simulation.environment.state.step,
        )
