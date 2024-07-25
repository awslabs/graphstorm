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

from numpy.testing import assert_array_equal
from pyspark.sql import functions as F, DataFrame, SparkSession

from pyspark.sql.types import ArrayType, IntegerType, LongType, StructField, StructType

from graphstorm_processing.data_transformations.dist_transformations import NoopTransformation


def test_noop_int_transformation(input_df: DataFrame, check_df_schema):
    """No-op transformation for integer columns"""
    col_name = "age"
    int_df = input_df.select(col_name).filter(F.col(col_name).isNotNull())
    noop_transfomer = NoopTransformation([col_name])

    transformed_df = noop_transfomer.apply(int_df)

    expected_values = [40, 20, 60, 40]

    check_df_schema(transformed_df)

    transformed_values = [row[col_name] for row in transformed_df.collect()]

    assert_array_equal(expected_values, transformed_values)


def test_noop_strvector_transformation(input_df: DataFrame, check_df_schema):
    """No-op transformation for string vector columns with separator"""
    col_name = "ratings"
    str_df = input_df.select(col_name)
    noop_transfomer = NoopTransformation([col_name], separator="|")

    transformed_df = noop_transfomer.apply(str_df)

    expected_values = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]

    check_df_schema(transformed_df)

    transformed_values = [row[col_name] for row in transformed_df.collect()]

    assert_array_equal(expected_values, transformed_values)


def test_noop_floatvector_transformation(spark: SparkSession, check_df_schema):
    """No-op transformation for numerical vector columns"""
    data = [([[10, 20]]), ([[30, 40]]), ([[50, 60]]), ([[70, 80]]), ([[90, 100]])]

    col_name = "feat"
    schema = StructType([StructField("feat", ArrayType(IntegerType(), True), True)])
    vec_df = spark.createDataFrame(data, schema=schema)

    noop_transfomer = NoopTransformation([col_name])

    transformed_df = noop_transfomer.apply(vec_df)

    expected_values = [[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]]

    check_df_schema(transformed_df)

    transformed_values = [row[col_name] for row in transformed_df.collect()]

    assert_array_equal(expected_values, transformed_values)


def test_noop_floatvector_truncation(spark: SparkSession, check_df_schema):
    """No-op transformation for numerical vector columns with truncation"""
    data = [([[10, 20]]), ([[30, 40]]), ([[50, 60]]), ([[70, 80]]), ([[90, 100]])]

    col_name = "feat"
    schema = StructType([StructField("feat", ArrayType(IntegerType(), True), True)])
    vec_df = spark.createDataFrame(data, schema=schema)

    noop_transfomer = NoopTransformation(
        [col_name],
        truncate_dim=1,
    )

    transformed_df = noop_transfomer.apply(vec_df)

    expected_values = [[10], [30], [50], [70], [90]]

    check_df_schema(transformed_df)

    transformed_values = [row[col_name] for row in transformed_df.collect()]

    assert_array_equal(expected_values, transformed_values)


def test_noop_largegint_transformation(spark: SparkSession, check_df_schema):
    """No-op transformation for long numerical columns"""
    large_int = 4 * 10**18
    data = [
        ([[large_int, large_int + 1]]),
        ([[large_int + 2, large_int + 3]]),
        ([[large_int + 4, large_int + 5]]),
        ([[large_int + 6, large_int + 7]]),
        ([[large_int + 8, large_int + 9]]),
    ]

    col_name = "feat"
    schema = StructType([StructField("feat", ArrayType(LongType(), True), True)])
    vec_df = spark.createDataFrame(data, schema=schema)

    noop_transfomer = NoopTransformation([col_name])

    transformed_df = noop_transfomer.apply(vec_df)

    check_df_schema(transformed_df)

    transformed_values = [row[col_name] for row in transformed_df.collect()]

    assert_array_equal([val[0] for val in data], transformed_values)
