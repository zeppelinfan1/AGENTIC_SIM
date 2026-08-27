import logging

from agentic_sim.workflows.logic.agents.train_agents_logic import TrainAgents


def main(**kwargs: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    task = TrainAgents(
        init_config=kwargs,
    )

    task.launch()
