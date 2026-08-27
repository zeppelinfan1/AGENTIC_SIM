from __future__ import annotations

import logging
import os

from collections.abc import Mapping
from typing import Any, Literal

from pyspark.sql import SparkSession

type RuntimeMode = Literal["auto", "databricks", "connect"]
type ResolvedRuntimeMode = Literal["databricks", "connect"]


def resolve_runtime_mode(
    runtime_mode: RuntimeMode = "auto",
) -> ResolvedRuntimeMode:
    if runtime_mode == "databricks":
        return "databricks"

    if runtime_mode == "connect":
        return "connect"

    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        return "databricks"

    return "connect"


def prepare_spark(
    *,
    spark_config: Mapping[str, str] | None = None,
    runtime_mode: RuntimeMode = "auto",
    connect_profile: str | None = None,
) -> SparkSession:
    logger = logging.getLogger(__name__)

    resolved_runtime = resolve_runtime_mode(runtime_mode)

    if resolved_runtime == "databricks":
        logger.info("Using Spark session from Databricks Runtime")

        spark = SparkSession.builder.getOrCreate()

    else:
        logger.info("Creating Spark session using Databricks Connect")

        from databricks.connect import DatabricksSession

        builder = DatabricksSession.builder

        if connect_profile is not None:
            builder = builder.profile(connect_profile)

        spark = builder.serverless().getOrCreate()

    for key, value in (spark_config or {}).items():
        logger.info("Setting Spark configuration: %s=%s", key, value)
        spark.conf.set(key, value)

    return spark


def prepare_dbutils(
    *,
    runtime_mode: RuntimeMode = "auto",
    connect_profile: str | None = None,
) -> Any:
    resolved_runtime = resolve_runtime_mode(runtime_mode)

    if resolved_runtime == "databricks":
        from databricks.sdk.runtime import dbutils

        return dbutils

    from databricks.sdk import WorkspaceClient

    if connect_profile is not None:
        workspace = WorkspaceClient(profile=connect_profile)
    else:
        workspace = WorkspaceClient()

    return workspace.dbutils
