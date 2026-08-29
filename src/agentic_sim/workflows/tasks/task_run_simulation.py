import logging

from agentic_sim.workflows.logic.simulation.simulation_engine_logic import (
    SimulationEngine,
)


def main(**kwargs: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    task = SimulationEngine(
        init_config=kwargs,
    )

    task.launch()
