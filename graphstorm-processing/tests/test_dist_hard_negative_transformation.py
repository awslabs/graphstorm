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

import pytest
from pyspark.sql import DataFrame, SparkSession
import numpy as np
from numpy.testing import assert_array_equal

from graphstorm_processing.constants import NODE_MAPPING_STR, NODE_MAPPING_INT
from graphstorm_processing.data_transformations.dist_transformations import (
    DistHardEdgeNegativeTransformation,
)


def test_hard_negative_example_list(spark: SparkSession, check_df_schema, tmp_path):
    # Input Data DataFrame
    data = [
        ("mark", "doctor", ["scientist"]),
        ("john", "scientist", ["engineer", "nurse"]),
        ("tara", "engineer", ["nurse", "doctor", "scientist"]),
        ("jen", "nurse", ["doctor"]),
    ]
    columns = ["src_type", "dst_type", "hard_negative"]
    input_df = spark.createDataFrame(data, schema=columns)

    # Mapping DataFrame
    mapping_data = [
        ("doctor", 0),
        ("scientist", 1),
        ("engineer", 2),
        ("nurse", 3),
    ]
    mapping_column = [NODE_MAPPING_STR, NODE_MAPPING_INT]
    mapping_df = spark.createDataFrame(mapping_data, schema=mapping_column)
    mapping_df.repartition(1).write.parquet(f"{tmp_path}/raw_id_mappings/dst_type/parquet")
    hard_node_mapping_dict = {
        "edge_type": "src_type:relation:dst_type",
        "mapping_path": f"{tmp_path}/raw_id_mappings/",
        "format_name": "parquet",
    }
    hard_negative_transformation = DistHardEdgeNegativeTransformation(
        ["hard_negative"],
        spark=spark,
        hard_node_mapping_dict=hard_node_mapping_dict,
        separator=None,
    )
    output_df = hard_negative_transformation.apply(input_df)
    check_df_schema(output_df)
    output_data = output_df.collect()

    # All the length should be the same as the maximum array.
    expected_output = [[1, -1, -1], [2, 3, -1], [3, 0, 1], [0, -1, -1]]

    for idx, row in enumerate(output_data):
        np.testing.assert_equal(row[0], expected_output[idx], err_msg=f"Row {idx} is not equal")


def test_hard_negative_example_str(spark: SparkSession, check_df_schema, tmp_path):
    # Input Data DataFrame
    data = [
        ("mark", "doctor", "scientist"),
        ("john", "scientist", "engineer;nurse"),
        ("tara", "engineer", "nurse;doctor;scientist"),
        ("jen", "nurse", "doctor"),
    ]
    columns = ["src_type", "dst_type", "hard_negative"]
    input_df = spark.createDataFrame(data, schema=columns)

    # Mapping DataFrame
    mapping_data = [
        ("doctor", 0),
        ("scientist", 1),
        ("engineer", 2),
        ("nurse", 3),
    ]
    mapping_column = [NODE_MAPPING_STR, NODE_MAPPING_INT]
    mapping_df = spark.createDataFrame(mapping_data, schema=mapping_column)
    mapping_df.repartition(1).write.parquet(f"{tmp_path}/raw_id_mappings/dst_type/parquet")
    hard_node_mapping_dict = {
        "edge_type": "src_type:relation:dst_type",
        "mapping_path": f"{tmp_path}/raw_id_mappings/",
        "format_name": "parquet",
    }
    hard_negative_transformation = DistHardEdgeNegativeTransformation(
        ["hard_negative"], spark=spark, hard_node_mapping_dict=hard_node_mapping_dict, separator=";"
    )
    output_df = hard_negative_transformation.apply(input_df)
    check_df_schema(output_df)
    output_data = output_df.collect()

    # All the length should be the same as the maximum array.
    expected_output = [[1, -1, -1], [2, 3, -1], [3, 0, 1], [0, -1, -1]]

    for idx, row in enumerate(output_data):
        np.testing.assert_equal(row[0], expected_output[idx], err_msg=f"Row {idx} is not equal")
