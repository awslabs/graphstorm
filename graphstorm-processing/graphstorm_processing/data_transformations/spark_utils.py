"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Utilities for Spark and SageMaker instance configuration.

We take some of the latest configuration code from the sagemaker-spark-container
that only exists for the Python 3.9/Spark 3.2 container
https://github.com/aws/sagemaker-spark-container/blob/4ef476fd535040f245def3d38c59fe43062e88a9/src/smspark/bootstrapper.py#L375
"""

import logging
import uuid
from typing import Optional, Tuple, Sequence

import psutil

import pyspark
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.util import VersionUtils
from graphstorm_processing import constants
from graphstorm_processing.constants import ExecutionEnv, FilesystemType, SPARK_HADOOP_VERSIONS

try:
    from smspark.bootstrapper import Bootstrapper
except ImportError:
    # smspark only exists on the SageMaker Docker image
    class Bootstrapper:  # type:ignore
        # pylint: disable=all
        def load_processing_job_config(self):
            return None

        def load_instance_type_info(self):
            return None


def create_spark_session(
    execution_env: ExecutionEnv, filesystem_type: FilesystemType
) -> SparkSession:
    """
    Create a SparkSession with the appropriate configuration for the execution context.

    Parameters
    ----------
    execution_env
        Whether or not this is being executed on a SageMaker instance.
    filesystem_type
        The filesystem type to use.

    Returns
    -------
    SparkSession
        The SparkSession.

    """
    spark_builder = (
        SparkSession.builder.appName("GSProcessing")
        .config("spark.hadoop.validateOutputSpecs", "false")
        .config("spark.logConf", "true")
    )

    if execution_env == ExecutionEnv.SAGEMAKER or execution_env == ExecutionEnv.LOCAL:
        # Set up Spark cluster config
        bootstraper = Bootstrapper()

        processing_job_config = bootstraper.load_processing_job_config()
        instance_type_info = bootstraper.load_instance_type_info()

        # For SM and local we set the driver/executor memory according to
        # the CPU and RAM resources detected on the driver, since we are running on metal
        spark_builder = _configure_spark_env_memory(
            spark_builder, processing_job_config, instance_type_info
        )

    major, minor = VersionUtils.majorMinorVersion(pyspark.__version__)
    hadoop_ver = SPARK_HADOOP_VERSIONS[f"{major}.{minor}"]
    # Only used for local testing and container execution
    if execution_env == ExecutionEnv.LOCAL and filesystem_type == FilesystemType.S3:
        logging.info("Setting up local Spark instance for S3 access...")
        spark_builder.config(
            "spark.jars.packages",
            f"org.apache.hadoop:hadoop-aws:{hadoop_ver},"
            f"org.apache.hadoop:hadoop-client:{hadoop_ver}",
        ).config("spark.jars.excludes", "com.google.guava:guava").config(
            "spark.executor.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"
        ).config(
            "spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"
        )

    spark = spark_builder.getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    logger = spark.sparkContext._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("py4j").setLevel(logger.Level.ERROR)
    spark_logger = logging.getLogger("py4j.java_gateway")
    spark_logger.setLevel(logging.ERROR)

    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()
    # This is needed to save RDDs which is the only way to write nested Dataframes into CSV format
    # hadoop_config.set(
    #     "mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter"
    # )
    # See https://aws.amazon.com/premiumsupport/knowledge-center/emr-timeout-connection-wait/
    hadoop_config.set("fs.s3.maxConnections", "5000")
    hadoop_config.set("fs.s3.maxRetries", "20")
    hadoop_config.set("fs.s3a.connection.maximum", "150")

    # Set up auth for local and EMR
    if execution_env != ExecutionEnv.SAGEMAKER and filesystem_type == FilesystemType.S3:
        hadoop_config.set(
            "fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
        )
        hadoop_config.set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        hadoop_config.set("fs.AbstractFileSystem.s3a.imp", "org.apache.hadoop.fs.s3a.S3A")
        spark.sparkContext.setSystemProperty("com.amazonaws.services.s3.enableV4", "true")

    return spark


def _configure_spark_env_memory(
    spark_builder: SparkSession.Builder,
    processing_job_config: Optional[dict],
    instance_type_info: Optional[dict],
) -> SparkSession.Builder:
    if processing_job_config and instance_type_info:
        instance_type = processing_job_config["ProcessingResources"]["ClusterConfig"][
            "InstanceType"
        ].replace("ml.", "")
        instance_type_details = instance_type_info[instance_type]
        instance_mem_mb = instance_type_details["MemoryInfo"]["SizeInMiB"]
        instance_cores = instance_type_details["VCpuInfo"]["DefaultVCpus"]
        logging.info(
            "Detected instance type: %s with total memory: %d MiB and total cores: %d",
            instance_type,
            instance_mem_mb,
            instance_cores,
        )
    else:
        instance_mem_mb = int(psutil.virtual_memory().total / (1024 * 1024))
        instance_cores = psutil.cpu_count(logical=True)
        logging.info(
            "Configuring Spark execution env using psutil values. Found total memory: %d MiB and total cores: %d",
            instance_mem_mb,
            instance_cores,
        )

    executor_cores = instance_cores
    executor_count_per_instance = int(instance_cores / executor_cores)

    driver_mem_mb = int(instance_mem_mb * constants.DRIVER_MEM_INSTANCE_MEM_RATIO)
    driver_mem_overhead_mb = int(driver_mem_mb * constants.DRIVER_MEM_OVERHEAD_RATIO)
    executor_mem_mb = int(
        (
            (instance_mem_mb * constants.EXECUTOR_MEM_INSTANCE_MEM_RATIO)
            / executor_count_per_instance
        )
        * (1 - constants.EXECUTOR_MEM_OVERHEAD_RATIO)
    )
    executor_mem_overhead_mb = int(executor_mem_mb * constants.EXECUTOR_MEM_OVERHEAD_RATIO)

    driver_max_result = min(driver_mem_overhead_mb, 4096)

    # Config to:
    # Allow CSV overwrite and configuring logging
    #   See https://aws.amazon.com/premiumsupport/knowledge-center/emr-timeout-connection-wait/
    # Improve memory utilization
    # Avoid timeout errors due to connection pool starving
    # Allow sending large results to driver
    spark_builder = (
        spark_builder.config("spark.driver.memory", f"{driver_mem_mb}m")
        .config("spark.driver.memoryOverhead", f"{driver_mem_overhead_mb}m")
        .config("spark.driver.maxResultSize", f"{driver_max_result}m")
        .config("spark.executor.memory", f"{executor_mem_mb}m")
        .config("spark.executor.memoryOverhead", f"{executor_mem_overhead_mb}m")
    )

    return spark_builder


def safe_rename_column(
    dataframe: DataFrame, old_colum_name: str, new_column_name: str
) -> Tuple[DataFrame, str]:
    """Safely rename a column in a dataframe.

    If the requested column to be renamed does not exist will log a warning.

    If the new name would be the same as another existing column, we modify the requested
    name and return the new modified value, along with the modified DataFrame.

    Parameters
    ----------
    dataframe : DataFrame
        The dataframe to be modified.
    old_colum_name : str
        The name of the column to be renamed.
    new_column_name : str
        The new name for the column.

    Returns
    -------
    tuple[DataFrame, str]
        The modified dataframe and the new column name.
    """
    if old_colum_name in dataframe.columns:
        if new_column_name in dataframe.columns:
            suffix = uuid.uuid4().hex[:8]
            logging.warning(
                "Column %s already exists in dataframe. Changing to %s.",
                new_column_name,
                new_column_name + suffix,
            )
            new_column_name = new_column_name + suffix
        # Rename column in dataframe.
        logging.debug("Renaming column %s to %s.", old_colum_name, new_column_name)
        dataframe = dataframe.withColumnRenamed(old_colum_name, new_column_name)
    else:
        logging.warning("Column %s not found in dataframe. Skipping renaming.", old_colum_name)
    return dataframe, new_column_name


def rename_multiple_cols(
    df: DataFrame, old_cols: Sequence[str], new_cols: Sequence[str]
) -> Tuple[DataFrame, Sequence[str]]:
    """Safely renames multiple columns at once. All columns not listed in the passed args are left as is.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame
    old_cols : Sequence[str]
        List of column names to change
    new_cols : Sequence[str]
        List of new column names.

    Returns
    -------
    Tuple[DataFrame, Sequence[str]]
        DataFrame with renamed columns, and a list of the new column names.
    """
    assert len(old_cols) == len(new_cols)
    safe_new_cols = []
    for old_name, new_name in zip(old_cols, new_cols):
        _, safe_new_name = safe_rename_column(df, old_name, new_name)
        safe_new_cols.append(safe_new_name)
    mapping = dict(zip(old_cols, safe_new_cols))
    return df.select([F.col(c).alias(mapping.get(c, c)) for c in df.columns]), safe_new_cols
