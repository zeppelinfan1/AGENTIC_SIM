from agentic_sim.workflows.task import Task

from agentic_sim.workflows.logic.agents.observation.observation_processor import (
    ObservationProcessor,
)
from agentic_sim.workflows.logic.simulation.build_simulation_logic import (
    BuildSimulation,
    SimulationContext,
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
        "num_steps",
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Runtime parameters
        self.is_dev_run = self.get_bool_parameter(
            "is_dev_run",
        )

        self.dev_catalog = self.init_config["dev_catalog"]

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

        self.num_steps = int(
            self.init_config["num_steps"],
        )

        self.logger.info(
            (
                "SimulationEngine parameters | "
                "is_dev_run=%s | "
                "dev_catalog=%s | "
                "num_agents=%s | "
                "num_markets=%s | "
                "num_institutions=%s | "
                "accounts_per_agent=%s | "
                "random_seed=%s | "
                "num_steps=%s"
            ),
            self.is_dev_run,
            self.dev_catalog,
            self.num_agents,
            self.num_markets,
            self.num_institutions,
            self.accounts_per_agent,
            self.random_seed,
            self.num_steps,
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
            ("Spark connection successful | " "catalog=%s | " "schema=%s"),
            current_environment["current_catalog"],
            current_environment["current_schema"],
        )

        # Build components
        self.build_simulation = BuildSimulation(
            num_agents=self.num_agents,
            num_markets=self.num_markets,
            num_institutions=self.num_institutions,
            accounts_per_agent=self.accounts_per_agent,
            random_seed=self.random_seed,
            spark=self.spark,
            logger=self.logger,
        )

        # Runtime processing components
        self.observation_processor = ObservationProcessor()

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

        agent_actions = []

        for agent in simulation.agents:

            # 1. Perceive current environment state
            observation = simulation.perception.perceive(
                agent=agent,
                environment=simulation.environment,
            )

            # 2. Process semantic perception into numerical input
            processed_observation = self.observation_processor.process(
                observation=observation,
            )

            # 3. Encode current perceived state
            state_embedding = agent.brain.encode(
                processed_observation.values,
            )

            # 4. Produce primitive action coordinates
            action_coordinates = agent.brain.actor(
                state_embedding,
            )

            agent_actions.append(
                (
                    agent,
                    action_coordinates,
                )
            )

            self.logger.info(
                (
                    "Agent decision generated | "
                    "step=%s | "
                    "agent_id=%s | "
                    "features=%s | "
                    "action_coordinates=%s"
                ),
                step,
                agent.config.agent_id,
                processed_observation.feature_names,
                action_coordinates.detach().tolist(),
            )

        # Future:
        #
        # 5. Interpret primitive action coordinates
        # 6. Resolve all requested actions against the environment
        # 7. Update environment state
        # 8. Calculate outcomes / rewards
        # 9. Store experience
        # 10. Perform learning

        self.logger.info(
            ("Simulation step completed | " "step=%s | " "agent_actions=%s"),
            step,
            len(agent_actions),
        )

    def run(self) -> None:

        self.logger.info("Starting simulation engine...")

        # Build the initial simulation world
        simulation = self.build_simulation.run()

        # Execute simulation through time
        for _ in range(
            self.num_steps,
        ):

            self._run_step(
                simulation=simulation,
            )

            simulation.environment.state.step += 1

        self.logger.info(
            ("Simulation engine completed | " "steps_completed=%s | " "final_step=%s"),
            self.num_steps,
            simulation.environment.state.step,
        )
