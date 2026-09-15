from agentic_sim.workflows.task import Task

from agentic_sim.workflows.logic.actions.action_candidate_provider import (
    ActionCandidateProvider,
)
from agentic_sim.workflows.logic.actions.action_candidate_value import (
    ValuedActionCandidates,
)
from agentic_sim.workflows.logic.actions.encoding.action_candidate_processor import (
    ActionCandidateProcessor,
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
from agentic_sim.workflows.logic.reward.positive_balance import (
    PositiveBalanceRewardMechanism,
)
from agentic_sim.workflows.logic.simulation.build_simulation_logic import (
    BuildSimulation,
    SimulationContext,
)
from agentic_sim.workflows.logic.actions.selection.greedy import (
    GreedyActionSelector,
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
        super().__init__(**kwargs)

        # Runtime parameters
        self.is_dev_run = self.get_bool_parameter("is_dev_run")
        self.dev_catalog = self.init_config["dev_catalog"]

        self.num_agents = int(self.init_config["num_agents"])
        self.num_markets = int(self.init_config["num_markets"])
        self.num_companies = int(self.init_config["num_companies"])
        self.num_institutions = int(self.init_config["num_institutions"])
        self.accounts_per_agent = int(self.init_config["accounts_per_agent"])
        self.random_seed = int(self.init_config["random_seed"])
        self.num_steps = int(self.init_config["num_steps"])

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

        # Validate Spark connection
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

        # Visibility / target discovery
        self.visible_field_extractor = VisibleFieldExtractor()

        self.target_candidate_provider = TargetCandidateProvider(
            visible_field_extractor=self.visible_field_extractor,
        )

        # Resolve stable target references against current environment
        self.target_reference_resolver = TargetReferenceResolver()

        # Determine feasible action magnitudes
        self.magnitude_constraint_resolver = MagnitudeConstraintResolver(
            target_reference_resolver=(self.target_reference_resolver),
        )

        self.magnitude_candidate_generator = MagnitudeCandidateGenerator()

        # Generate concrete target + direction + magnitude actions
        self.action_candidate_provider = ActionCandidateProvider(
            magnitude_constraint_resolver=(self.magnitude_constraint_resolver),
            magnitude_candidate_generator=(self.magnitude_candidate_generator),
        )

        # Convert action candidates into neural-network features
        self.action_candidate_processor = ActionCandidateProcessor()

        # Action selection
        self.action_selector = GreedyActionSelector()

        # Current V1 reward definition
        self.reward_mechanism = PositiveBalanceRewardMechanism(
            reward_scale=1.0,
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

        agent_decisions = []

        for agent in simulation.agents:

            actor_id = agent.config.agent_id

            # 1. Observe the current world.
            observation = simulation.perception.perceive(
                agent=agent,
                environment=simulation.environment,
            )

            # 2. Find visible fields that can participate in actions.
            target_candidates = self.target_candidate_provider.get_candidates(
                observation=observation,
                environment=simulation.environment,
            )

            # 3. Build concrete action candidates.
            action_candidates = self.action_candidate_provider.get_candidates(
                actor_id=actor_id,
                target_candidates=target_candidates,
                environment=simulation.environment,
            )

            # 4. Convert the current observation into model features.
            processed_observation = self.observation_processor.process(
                observation=observation,
            )

            # 5. Convert every action candidate into model features.
            processed_action_candidates = self.action_candidate_processor.process_many(
                candidates=action_candidates,
                observation=observation,
            )

            # 6. Encode the state and actions, then predict a value
            #    for every available action candidate.
            (
                state_embedding,
                action_embeddings,
                predicted_values,
            ) = agent.brain.evaluate_opportunities(
                observation_features=(processed_observation.values),
                action_features=(processed_action_candidates.values),
            )

            # 7. Keep each semantic candidate aligned with its
            #    predicted value.
            valued_action_candidates = ValuedActionCandidates(
                candidates=(processed_action_candidates.candidates),
                predicted_values=predicted_values,
            )

            # 8. Select the candidate with the highest predicted value.
            selected_action = self.action_selector.select(
                valued_candidates=valued_action_candidates,
            )

            agent_decisions.append(
                (
                    agent,
                    state_embedding,
                    action_embeddings,
                    valued_action_candidates,
                    selected_action,
                )
            )

            if selected_action is None:
                self.logger.info(
                    (
                        "No action selected | "
                        "step=%s | "
                        "agent_id=%s | "
                        "reason=no_candidates"
                    ),
                    step,
                    actor_id,
                )

            else:
                candidate = selected_action.candidate
                self.logger.info(
                    (
                        "Action selected | "
                        "step=%s | "
                        "agent_id=%s | "
                        "candidate_index=%s | "
                        "object_type=%s | "
                        "object_id=%s | "
                        "field=%s | "
                        "direction=%s | "
                        "magnitude=%s | "
                        "predicted_value=%s"
                    ),
                    step,
                    actor_id,
                    selected_action.candidate_index,
                    candidate.target.object_type,
                    candidate.target.object_id,
                    candidate.target.field_name,
                    candidate.direction.value,
                    candidate.magnitude,
                    selected_action.predicted_value,
                )

            self.logger.debug(
                (
                    "Valued action candidates | "
                    "step=%s | "
                    "agent_id=%s | "
                    "candidates=%s"
                ),
                step,
                actor_id,
                [
                    {
                        "target": {
                            "object_type": (candidate.target.object_type),
                            "object_id": (candidate.target.object_id),
                            "field_name": (candidate.target.field_name),
                        },
                        "direction": (candidate.direction.value),
                        "magnitude": candidate.magnitude,
                        "predicted_value": (
                            valued_action_candidates.predicted_values[
                                candidate_index
                            ].item()
                        ),
                    }
                    for (
                        candidate_index,
                        candidate,
                    ) in enumerate(valued_action_candidates.candidates)
                ],
            )

        # Reward cannot be calculated yet.
        #
        # The next stages are:
        #
        # 1. Select one action per agent.
        # 2. Capture each agent's pre-action reward state.
        # 3. Execute / jointly resolve the selected actions.
        # 4. Capture each agent's post-action reward state.
        # 5. Calculate reward.
        # 6. Store the transition.
        # 7. Train the value network.

        self.logger.info(
            ("Simulation step completed | " "step=%s | " "agent_decisions=%s"),
            step,
            len(agent_decisions),
        )

    def run(
        self,
    ) -> None:

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
