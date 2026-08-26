from agentic_sim.workflows.logic.simulation.simulation_engine import SimulationEngine
from agentic_sim.config.settings import Settings


def main():
    config = Settings()
    engine = SimulationEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
