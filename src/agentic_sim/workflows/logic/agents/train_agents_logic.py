from agentic_sim.workflows.task import Task


class TrainAgents(Task):

    def run(self) -> None:
        is_dev_run = self.get_bool_parameter("is_dev_run")
        dev_catalog = self.init_config["dev_catalog"]

        self.logger.info(
            "TrainAgents parameters | is_dev_run=%s | dev_catalog=%s",
            is_dev_run,
            dev_catalog,
        )

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
