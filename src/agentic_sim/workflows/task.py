from __future__ import annotations

import logging
import time

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from pyspark.sql import SparkSession

from agentic_sim.workflows.runtime import (
    RuntimeMode,
    prepare_dbutils,
    prepare_spark,
)


class Task(ABC):

    NAMED_PARAMETER_KEYS: tuple[str, ...] = ()
    REQUIRED_PARAMETER_KEYS: tuple[str, ...] | None = None

    SPARK_CONFIG: Mapping[str, str] = {}

    def __init__(
        self,
        *,
        init_config: Mapping[str, str] | None = None,
        spark_config: Mapping[str, str] | None = None,
        spark: SparkSession | None = None,
        runtime_mode: RuntimeMode = "auto",
        connect_profile: str | None = None,
    ) -> None:

        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        self.runtime_mode: RuntimeMode = runtime_mode
        self.connect_profile = connect_profile

        raw_config = dict(init_config or {})

        self.init_config = {
            key: raw_config[key]
            for key in self.NAMED_PARAMETER_KEYS
            if key in raw_config
        }

        self._validate_parameters()

        self.spark_config = {
            **self.SPARK_CONFIG,
            **dict(spark_config or {}),
        }

        self.spark = spark or prepare_spark(
            spark_config=self.spark_config,
            runtime_mode=self.runtime_mode,
            connect_profile=self.connect_profile,
        )

        self.dbutils = prepare_dbutils(
            runtime_mode=self.runtime_mode,
            connect_profile=self.connect_profile,
        )

    def launch(self) -> Any:
        start_time = time.perf_counter()

        self.logger.info(
            "Launching task: %s",
            self.__class__.__name__,
        )

        try:
            result = self.run()

        except Exception:
            self.logger.exception(
                "Task failed: %s",
                self.__class__.__name__,
            )
            raise

        elapsed = time.perf_counter() - start_time

        self.logger.info(
            "Task completed successfully: %s [%.2fs]",
            self.__class__.__name__,
            elapsed,
        )

        return result

    def _prepare_runtime(self) -> None:
        if self._spark is None:
            self._spark = prepare_spark(
                spark_config=self.spark_config,
                runtime_mode=self.runtime_mode,
                connect_profile=self.connect_profile,
            )

        self.dbutils = prepare_dbutils(
            runtime_mode=self.runtime_mode,
            connect_profile=self.connect_profile,
        )

    def _validate_parameters(self) -> None:

        required_keys = (
            self.NAMED_PARAMETER_KEYS
            if self.REQUIRED_PARAMETER_KEYS is None
            else self.REQUIRED_PARAMETER_KEYS
        )

        missing_parameters = [
            key for key in required_keys if key not in self.init_config
        ]

        if missing_parameters:
            raise ValueError(f"Missing required task parameters: {missing_parameters}")

    def get_parameter(
        self,
        key: str,
        default: str | None = None,
    ) -> str | None:
        return self.init_config.get(key, default)

    def get_bool_parameter(
        self,
        key: str,
        default: bool = False,
    ) -> bool:
        value = self.init_config.get(key)

        if value is None:
            return default

        return value.lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    @abstractmethod
    def run(self) -> Any:
        raise NotImplementedError
