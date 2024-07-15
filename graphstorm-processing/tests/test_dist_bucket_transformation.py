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

from pyspark.sql import DataFrame, SparkSession
import numpy as np
from numpy.testing import assert_array_equal

from graphstorm_processing.data_transformations.dist_transformations import (
    DistBucketNumericalTransformation,
)


def test_bucket_numerical_without_missing(user_df: DataFrame):
    bucket_transformation = DistBucketNumericalTransformation(["age"], [22, 33], 3, 0, "none")

    output_df = bucket_transformation.apply(user_df)

    assert output_df.select("age").distinct().count() == 3


def test_bucket_numerical_example(spark: SparkSession, check_df_schema):
    data = [("mark", 0.0, None), ("john", 15.0, 10000), ("tara", 26.0, 20000), ("jen", 40.0, 10000)]

    columns = ["name", "age", "salary"]
    input_df = spark.createDataFrame(data, schema=columns)

    low = 10.0
    high = 30.0
    bucket_cnt = 4
    window_size = 10.0  # range is 10 ~ 15; 15 ~ 20; 20 ~ 25; 25 ~ 30

    bucket_transformation = DistBucketNumericalTransformation(
        ["age"], [low, high], bucket_cnt, window_size, "none"
    )

    output_df = bucket_transformation.apply(input_df)

    check_df_schema(output_df)

    out_rows = output_df.collect()

    expected_vals = np.array(
        [[1.0, 0.0, 0.0, 0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )

    for i, row in enumerate(out_rows):
        assert_array_equal(row["age"], expected_vals[i, :], err_msg=f"Row {i} is not equal")


def test_bucket_numerical_second_example(spark: SparkSession):
    data = [("john", 21.0, None), ("tim", 31.0, 10000), ("maggie", 55.0, 20000)]

    columns = ["name", "age", "salary"]
    input_df = spark.createDataFrame(data, schema=columns)

    low = 0.0
    high = 100.0
    bucket_cnt = 10
    window_size = 5.0

    bucket_transformation = DistBucketNumericalTransformation(
        ["age"], [low, high], bucket_cnt, window_size, "none"
    )

    output_df = bucket_transformation.apply(input_df)

    out_rows = output_df.collect()

    expected_vals = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    for i, row in enumerate(out_rows):
        assert_array_equal(row["age"], expected_vals[i, :], err_msg=f"Row {i} is not equal")


def test_bucket_numerical_third_example(spark: SparkSession):
    data = [("john", 21.0, None), ("tim", 31.0, 10000), ("maggie", 55.0, 20000)]

    columns = ["name", "age", "salary"]
    input_df = spark.createDataFrame(data, schema=columns)

    low = 0.0
    high = 100.0
    bucket_cnt = 10

    bucket_transformation = DistBucketNumericalTransformation(["age"], [low, high], bucket_cnt)

    output_df = bucket_transformation.apply(input_df)

    out_rows = output_df.collect()

    expected_vals = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    for i, row in enumerate(out_rows):
        assert_array_equal(row["age"], expected_vals[i, :], err_msg=f"Row {i} is not equal")
