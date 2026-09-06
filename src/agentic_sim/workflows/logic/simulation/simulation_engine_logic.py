from agentic_sim.workflows.task import Task

from agentic_sim.workflows.logic.actions.candidates.action_candidate_provider import (
    ActionCandidateProvider,
)
from agentic_sim.workflows.logic.actions.magnitude.magnitude_candidate_generator import (
    MagnitudeCandidateGenerator,
)
from agentic_sim.workflows.logic.actions.magnitude.magnitude_constraint_resolver import (
    MagnitudeConstraintResolver,
)
from agentic_sim.workflows.logic.actions.targets.target_candidate_provider import (
    TargetCandidateProvider,
)
from agentic_sim.workflows.logic.actions.targets.target_reference_resolver import (
    TargetReferenceResolver,
)
from agentic_sim.workflows.logic.agents.observation.observation_processor import (
    ObservationProcessor,
)
from agentic_sim.workflows.logic.environment.perception.extractor import (
    VisibleFieldExtractor,
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
        "num_companies",
        "num_institutions",
        "accounts_per_agent",
        "random_seed",
        "num_steps",
    )

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(
            **kwargs,
        )

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

        self.num_companies = int(
            self.init_config["num_companies"],
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
                "num_companies=%s | "
                "num_institutions=%s | "
                "accounts_per_agent=%s | "
                "random_seed=%s | "
                "num_steps=%s"
            ),
            self.is_dev_run,
            self.dev_catalog,
            self.num_agents,
            self.num_markets,
            self.num_companies,
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
            raise RuntimeError(("Spark environment query " "returned no rows."))

        self.logger.info(
            ("Spark connection successful | " "catalog=%s | " "schema=%s"),
            current_environment["current_catalog"],
            current_environment["current_schema"],
        )

        # Build simulation runtime
        self.build_simulation = BuildSimulation(
            num_agents=self.num_agents,
            num_markets=self.num_markets,
            num_companies=self.num_companies,
            num_institutions=self.num_institutions,
            accounts_per_agent=self.accounts_per_agent,
            random_seed=self.random_seed,
            spark=self.spark,
            logger=self.logger,
        )

        # Observation processing
        self.observation_processor = ObservationProcessor()

        # Shared target field visibility logic
        self.visible_field_extractor = VisibleFieldExtractor()

        # Target discovery
        self.target_candidate_provider = TargetCandidateProvider(
            visible_field_extractor=(self.visible_field_extractor),
        )

        # Stable action-target resolution
        self.target_reference_resolver = TargetReferenceResolver()

        # Magnitude physics
        self.magnitude_constraint_resolver = MagnitudeConstraintResolver(
            target_reference_resolver=(self.target_reference_resolver),
        )

        self.magnitude_candidate_generator = MagnitudeCandidateGenerator()

        # Concrete action opportunity generation
        self.action_candidate_provider = ActionCandidateProvider(
            magnitude_constraint_resolver=(self.magnitude_constraint_resolver),
            magnitude_candidate_generator=(self.magnitude_candidate_generator),
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

        agent_opportunity_sets = []

        for agent in simulation.agents:

            actor_id = agent.config.agent_id

            # 1. Perceive current environment state
            observation = simulation.perception.perceive(
                agent=agent,
                environment=(simulation.environment),
            )

            # 2. Discover visible environment fields
            #    eligible for at least one primitive
            #    transformation.
            target_candidates = self.target_candidate_provider.get_candidates(
                observation=observation,
                environment=(simulation.environment),
            )

            # 3. Expand target fields into concrete
            #    target + direction + magnitude candidates.
            action_candidates = self.action_candidate_provider.get_candidates(
                actor_id=actor_id,
                target_candidates=(target_candidates),
                environment=(simulation.environment),
            )

            # 4. Process semantic perception into
            #    numerical model input.
            processed_observation = self.observation_processor.process(
                observation=observation,
            )

            # 5. Encode the current perceived state.
            #
            # Candidate evaluation is intentionally not
            # connected yet. The next architecture stage
            # will combine this embedding with candidate
            # representations.
            state_embedding = agent.brain.encode(
                processed_observation.values,
            )

            agent_opportunity_sets.append(
                (
                    agent,
                    state_embedding,
                    tuple(
                        action_candidates,
                    ),
                )
            )

            self.logger.info(
                (
                    "Agent opportunity set generated | "
                    "step=%s | "
                    "agent_id=%s | "
                    "features=%s | "
                    "target_candidates=%s | "
                    "action_candidates=%s"
                ),
                step,
                actor_id,
                processed_observation.feature_names,
                len(
                    target_candidates,
                ),
                len(
                    action_candidates,
                ),
            )

            self.logger.debug(
                (
                    "Agent action candidates | "
                    "step=%s | "
                    "agent_id=%s | "
                    "candidates=%s"
                ),
                step,
                actor_id,
                [
                    {
                        "object_type": (candidate.target.object_type),
                        "object_id": (candidate.target.object_id),
                        "field_name": (candidate.target.field_name),
                        "direction": (candidate.direction.value),
                        "magnitude": (candidate.magnitude),
                    }
                    for candidate in action_candidates
                ],
            )

        # Next development stage:
        #
        # 6. Encode every ActionCandidate
        # 7. Score state + candidate combinations
        # 8. Add explicit no-op option
        # 9. Explore/select one candidate per agent
        # 10. Collect all requested actions
        # 11. Jointly resolve requests against environment
        # 12. Measure objective consequences
        # 13. Form subjective rewards
        # 14. Store experience
        # 15. Perform reinforcement-learning updates

        self.logger.info(
            ("Simulation step completed | " "step=%s | " "agent_opportunity_sets=%s"),
            step,
            len(
                agent_opportunity_sets,
            ),
        )

    def run(
        self,
    ) -> None:

        self.logger.info("Starting simulation engine...")

        simulation = self.build_simulation.run()

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
