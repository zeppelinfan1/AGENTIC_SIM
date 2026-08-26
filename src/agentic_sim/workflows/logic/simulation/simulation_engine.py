class SimulationEngine:

    def __init__(self, config):
        self.config = config
        self.environment = ...
        self.agents = ...
        self.governance = ...

    def run(self):
        for episode in range(self.config.num_episodes):
            self.run_episode()

    def run_episode(self):
        """Example of what run_episode might look like:

        for step in range(self.config.steps_per_episode):

            observations = self.environment.observe()

            agent_actions = self.agents.act(observations)

            governance_actions = self.governance.respond(
                observations,
                agent_actions,
            )

            results = self.environment.step(
                agent_actions,
                governance_actions,
            )

            self.agents.learn(results)
            self.governance.learn(results)
        """
        ...
