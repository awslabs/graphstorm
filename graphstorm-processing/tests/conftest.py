"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import logging
import tempfile
from typing import Iterator

import numpy as np
import pytest
from pyspark.sql import SparkSession, DataFrame
from pyarrow import parquet as pq
import pyarrow

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["MODEL_PATH"] = "./.pretrain/models"
os.environ["DEPLOYMENT_STAGE"] = "dev"

_ROOT = os.path.abspath(os.path.dirname(__file__))


def in_docker():
    """Returns: True if running in a Docker container, else False"""
    with open("/proc/1/cgroup", "rt", encoding="utf-8") as ifh:
        return "docker" in ifh.read()


def suppress_py4j_logging():
    """Suppress extra Spark logging"""
    logger = logging.getLogger("py4j")
    logger.setLevel(logging.WARN)


@pytest.fixture(scope="session", autouse=True)
def temp_output_root():
    """Create a root temporary directory for output files.

    Individual tests create per-test temporary directories
    and output under this root directory.

    Individual tests are responsible for deleting their output
    under the directory, will raise an OSError if the directory
    is not empty at the end of testing.
    """
    yield os.mkdir(os.path.join(_ROOT, "resources/test_output/"))
    os.rmdir(os.path.join(_ROOT, "resources/test_output/"))


@pytest.fixture(scope="session", name="spark")
def spark_fixture() -> Iterator[SparkSession]:
    """Create the main SparkContext we use throughout the tests"""
    spark_context = (
        SparkSession.builder.master("local[4]")
        .appName("local-testing-pyspark-context")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    suppress_py4j_logging()
    yield spark_context

    spark_context.stop()


@pytest.fixture(scope="session")
def user_df(spark: SparkSession) -> DataFrame:
    """Re-usable user node df"""
    data_path = os.path.join(_ROOT, "resources/small_heterogeneous_graph/nodes/user.csv")
    cat_df = spark.read.csv(data_path, header=True, inferSchema=True)
    return cat_df


@pytest.fixture(scope="session")
def input_df(spark: SparkSession) -> DataFrame:
    """Re-usable input DF"""
    data = [
        ("mark", 40, None, "1|2"),
        ("john", None, 10000, "3|4"),
        ("tara", 20, 20000, "5|6"),
        ("jen", 60, 10000, "7|8"),
        ("kate", 40, 40000, "9|10"),
    ]

    columns = ["name", "age", "salary", "ratings"]
    complex_df = spark.createDataFrame(data, schema=columns)

    return complex_df


@pytest.fixture(scope="session")
def check_df_schema():
    """Function to ensure DF schema conforms to dist processing assumptions"""

    def ensure_df_conforms_to_schema_when_written(df: DataFrame):
        # Ensure written DF is either flat numerical values or a single numerical list col
        # as that is what the partitioning pipeline expects

        accepted_datatypes = {np.float32, np.float64, np.int32, np.int64}
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_file_path = os.path.join(tmpdirname, "tmp_file")
            df.coalesce(1).write.parquet(tmp_file_path)

            pq_table = pq.read_table(tmp_file_path)

            table_schema = pq_table.schema

            # If there's only one column and is a list,
            # its elements must be in the accepted data types
            if len(table_schema.types) == 1 and isinstance(table_schema.types[0], pyarrow.ListType):
                assert table_schema.types[0].value_type.to_pandas_dtype() in accepted_datatypes
            else:
                # Otherwise, ensure every plain column is of a numerical type
                for schema_type in table_schema.types:
                    assert schema_type.to_pandas_dtype() in accepted_datatypes

    return ensure_df_conforms_to_schema_when_written
