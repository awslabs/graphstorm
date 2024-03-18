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

from typing import Tuple, Iterator
import os
import pytest
import pandas as pd
import tempfile

import mock
from numpy.testing import assert_array_equal
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructField, StructType, StringType, ArrayType

from graphstorm_processing.data_transformations.dist_transformations import (
    DistCategoryTransformation,
    DistMultiCategoryTransformation,
)
from graphstorm_processing.constants import RARE_CATEGORY


pytestmark = pytest.mark.usefixtures("spark")
_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(name="multi_cat_df_and_separator")
def multi_cat_df_and_separator_fixture(
    spark: SparkSession, separator=","
) -> Iterator[Tuple[DataFrame, str]]:
    """Gneerate multi-category df, yields the DF and its separator"""
    data = [
        (f"Actor{separator}Director",),
        (f"Director{separator}Writer",),
        ("Director",),
        (f"Actor{separator}",),
        (f"Director{separator}Writer{separator}Actor",),
    ]

    col_name = "credit"
    schema = StructType([StructField(col_name, StringType(), True)])
    multi_cat_df = spark.createDataFrame(data, schema=schema)

    yield multi_cat_df, separator


@mock.patch(
    (
        "graphstorm_processing.data_transformations."
        "dist_transformations.dist_category_transformation.MAX_CATEGORIES_PER_FEATURE"
    ),
    3,
)
def test_limited_category_transformation(user_df):
    """Test single-cat transformation with limited categories"""
    dist_category_transformation = DistCategoryTransformation(["occupation"])

    transformed_df = dist_category_transformation.apply(user_df)
    group_counts = (
        transformed_df.groupBy("occupation").count().orderBy("count", ascending=False).collect()
    )

    expected_counts = [2, 2, 1]  # 2 for student, 2 for two uniques grouped, 1 more unique
    for row, expected_count in zip(group_counts, expected_counts):
        assert row["count"] == expected_count


def test_all_categories_transformation(user_df, check_df_schema):
    """Test single-cat transformation with all categories"""
    dist_category_transformation = DistCategoryTransformation(["occupation"])

    transformed_df = dist_category_transformation.apply(user_df)

    check_df_schema(transformed_df)

    transformed_distinct_values = transformed_df.distinct().count()

    assert transformed_distinct_values == 4


def test_category_transformation_with_null_values(spark: SparkSession):
    """Test null value transformation"""
    data = [
        ("mark", "student", "male"),
        ("john", "actor", "male"),
        ("tara", None, "female"),
        ("jen", "doctor", None),
        ("kate", "", "female"),
    ]

    columns = ["name", "occupation", "gender"]
    df = spark.createDataFrame(data, schema=columns)
    dist_category_transformation = DistCategoryTransformation(["occupation"])

    transformed_df = dist_category_transformation.apply(df)

    transformed_distinct_values = transformed_df.distinct().count()

    assert transformed_distinct_values == 4


def test_multiple_categories_transformation(user_df):
    """Test transforming multiple cat columns"""
    dist_category_transformation = DistCategoryTransformation(["occupation", "gender"])

    transformed_df = dist_category_transformation.apply(user_df)

    occupation_distinct_values = transformed_df.select("occupation").distinct().count()
    gender_distinct_values = transformed_df.select("gender").distinct().count()

    assert occupation_distinct_values == 4
    assert gender_distinct_values == 3


def test_multi_category_transformation(multi_cat_df_and_separator, check_df_schema):
    """Test transforming multi-category column"""
    df, separator = multi_cat_df_and_separator
    col_name = df.columns[0]

    dist_multi_transformation = DistMultiCategoryTransformation([col_name], separator)

    transformed_df = dist_multi_transformation.apply(df)

    check_df_schema(transformed_df)

    expected_values = [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]

    expected_value_map = {"Director": 0, "Actor": 1, "Writer": 2}

    assert expected_value_map == dist_multi_transformation.value_map

    transformed_values = [row[col_name] for row in transformed_df.collect()]

    assert_array_equal(expected_values, transformed_values)


@mock.patch(
    (
        "graphstorm_processing.data_transformations."
        "dist_transformations.dist_category_transformation.MAX_CATEGORIES_PER_FEATURE"
    ),
    2,
)
def test_multi_category_limited_categories(multi_cat_df_and_separator):
    """Test transforming multi-category column with limited categories"""
    df, separator = multi_cat_df_and_separator
    col_name = df.columns[0]

    dist_multi_transformation = DistMultiCategoryTransformation([col_name], separator)

    transformed_df = dist_multi_transformation.apply(df)

    expected_values = [[1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

    expected_value_map = {RARE_CATEGORY: 0, "Director": 1}

    assert expected_value_map == dist_multi_transformation.value_map

    transformed_values = [row[col_name] for row in transformed_df.collect()]

    assert_array_equal(expected_values, transformed_values)


def test_csv_input_categorical(spark: SparkSession, check_df_schema):
    data_path = os.path.join(_ROOT, "resources/multi_num_numerical/multi_num.csv")
    long_vector_df = spark.read.csv(data_path, sep=",", header=True)
    dist_categorical_transormation = DistCategoryTransformation(cols=["id"])

    transformed_df = dist_categorical_transormation.apply(long_vector_df)
    check_df_schema(transformed_df)
    transformed_rows = transformed_df.collect()
    expected_rows = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    for row, expected_row in zip(transformed_rows, expected_rows):
        assert row["id"] == expected_row


def test_csv_input_multi_categorical(spark: SparkSession, check_df_schema):
    data_path = os.path.join(_ROOT, "resources/multi_num_numerical/multi_num.csv")
    long_vector_df = spark.read.csv(data_path, sep=",", header=True)
    dist_categorical_transormation = DistMultiCategoryTransformation(cols=["feat"], separator=";")

    transformed_df = dist_categorical_transormation.apply(long_vector_df)
    check_df_schema(transformed_df)
    transformed_rows = transformed_df.collect()
    expected_rows = []
    for i in range(5):
        expected_rows.append([1] * 100)
    for row, expected_row in zip(transformed_rows, expected_rows):
        assert row["feat"] == expected_row


def test_parquet_input_multi_categorical(spark: SparkSession, check_df_schema):
    # Define the schema for the DataFrame
    schema = StructType([StructField("names", ArrayType(StringType()), True)])

    # Sample data with arrays of strings
    data = [
        (["Alice", "Alicia"],),
        (["Bob", "Bobby"],),
        (["Cathy", "Catherine"],),
        (["David", "Dave"],),
    ]

    # Create a DataFrame using the sample data and the defined schema
    df = spark.createDataFrame(data, schema)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Define the path for the Parquet file
        parquet_path = f"{tmpdirname}/people_name.parquet"

        # Write the DataFrame to a Parquet file
        df.write.mode("overwrite").parquet(parquet_path)

        # Read the Parquet file into a DataFrame
        df_parquet = spark.read.parquet(parquet_path)

        # Show the DataFrame loaded from the Parquet file
        dist_categorical_transormation = DistMultiCategoryTransformation(
            cols=["names"], separator=None
        )

        transformed_df = dist_categorical_transormation.apply(df_parquet)
        check_df_schema(transformed_df)
        transformed_rows = transformed_df.collect()
        expected_rows = [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ]
        for row, expected_row in zip(transformed_rows, expected_rows):
            assert row["names"] == expected_row
