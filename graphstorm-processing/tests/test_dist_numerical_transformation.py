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

import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    DoubleType,
    StructField,
    StructType,
    StringType,
    LongType,
)
from scipy.special import erfinv  # pylint: disable=no-name-in-module

from graphstorm_processing.data_transformations.dist_transformations import (
    DistNumericalTransformation,
    DistMultiNumericalTransformation,
)

_ROOT = os.path.abspath(os.path.dirname(__file__))


@pytest.fixture(scope="function", name="multi_num_df_with_missing")
def multi_num_df_with_missing_fixture(spark: SparkSession):
    """Create a multi-numerical text DataFrame with missing data"""
    data = [("1|10",), ("5|2",), ("3|",), ("1|3",), ("5|2",)]

    schema = StructType([StructField("ratings", StringType(), True)])
    df = spark.createDataFrame(data, schema=schema)

    yield df


@pytest.fixture(scope="function", name="multi_num_df_without_missing")
def multi_num_df_without_missing_fixture(spark: SparkSession):
    """Create a multi-numerical text DataFrame without missing data"""
    data = [("1|10",), ("5|2",), ("3|2",), ("1|3",), ("5|2",)]

    schema = StructType([StructField("ratings", StringType(), True)])
    df = spark.createDataFrame(data, schema=schema)

    yield df


@pytest.fixture(scope="function", name="multi_num_df_with_vectors")
def multi_num_df_with_vectors_fixture(spark: SparkSession):
    """Create a multi-numerical text DataFrame without missing data"""
    data = [([1, 2],), ([3, 4],), ([5, 6],), ([7, 8],), ([9, 10],)]

    schema = StructType([StructField("ratings", ArrayType(LongType()), True)])
    df = spark.createDataFrame(data, schema=schema)

    yield df


def test_numerical_transformation_with_mode_imputer(input_df: DataFrame):
    """Test numerical mode imputer"""
    dist_numerical_transformation = DistNumericalTransformation(
        ["salary", "age"], imputer="most_frequent", normalizer="none"
    )

    transformed_df = dist_numerical_transformation.apply(input_df)

    transformed_rows = transformed_df.collect()

    for row in transformed_rows:
        if row["name"] == "mark":
            assert row["salary"] == 10000
        elif row["name"] == "john":
            assert row["age"] == 40
        else:
            assert row["salary"] in {10000, 20000, 40000}
            assert row["age"] in {20, 40, 60}


def test_numerical_transformation_with_minmax_scaler(input_df: DataFrame):
    """Test numerical min-max normalizer"""
    no_na_df = input_df.na.fill(0)
    dist_numerical_transformation = DistNumericalTransformation(
        ["age", "salary"], imputer="none", normalizer="min-max"
    )

    transformed_df = dist_numerical_transformation.apply(no_na_df)

    transformed_rows = transformed_df.collect()

    for row in transformed_rows:
        if row["name"] == "kate":
            assert row["salary"] == 1.0
        elif row["name"] == "mark":
            assert row["salary"] == 0.0
        else:
            assert row["salary"] < 1.0 and row["salary"] > 0.0


def test_numerical_transformation_without_transformation(input_df: DataFrame, check_df_schema):
    """Test numerical transformation without any transformation applied"""
    no_na_df = input_df.select(["age", "salary"]).na.fill(0)

    dist_numerical_transformation = DistNumericalTransformation(
        ["age", "salary"], imputer="none", normalizer="none"
    )

    transformed_df = dist_numerical_transformation.apply(no_na_df)

    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()

    expected_salaries = [0, 10000, 20000, 10000, 40000]

    for row, expected_salary in zip(transformed_rows, expected_salaries):
        assert row["salary"] == expected_salary


@pytest.mark.parametrize("out_dtype", ["float32", "float64"])
def test_numerical_min_max_transformation_precision(
    spark: SparkSession, check_df_schema, out_dtype
):
    """Test min-max transformation with requested out_dtype"""
    # Adjust the number to be an integer
    high_precision_integer = 1.2345678901234562
    data = [(high_precision_integer,)]
    schema = StructType([StructField("age", FloatType(), True)])
    input_df = spark.createDataFrame(data, schema=schema)

    dist_numerical_transformation = DistNumericalTransformation(
        ["age"], imputer="none", normalizer="min-max", out_dtype=out_dtype
    )

    transformed_df = dist_numerical_transformation.apply(input_df)
    check_df_schema(transformed_df)
    column_data_type = [
        field.dataType for field in transformed_df.schema.fields if field.name == "age"
    ][0]
    if out_dtype == "float32":
        assert isinstance(column_data_type, FloatType), "The column 'age' is not of type FloatType."
    elif out_dtype == "float64":
        assert isinstance(
            column_data_type, DoubleType
        ), "The column 'age' is not of type DoubleType."


def test_numerical_transformation_with_median_imputer_and_std_norm(
    input_df: DataFrame, check_df_schema
):
    """Test numerical standard normalizer with median imputation"""
    input_df = input_df.select(["age", "salary"])
    dist_numerical_transformation = DistNumericalTransformation(
        ["age", "salary"], imputer="median", normalizer="standard"
    )

    transformed_df = dist_numerical_transformation.apply(input_df)

    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()

    expected_imputed_std_ages = [0.2, 0.2, 0.1, 0.3, 0.2]

    for row, expected_val in zip(transformed_rows, expected_imputed_std_ages):
        assert_almost_equal(row["age"], expected_val, decimal=7)


def test_multi_numerical_transformation_without_norm_and_imputer(input_df: DataFrame):
    """Test multi-numerical transformation without any transformation applied"""
    dist_multi_numerical_transormation = DistMultiNumericalTransformation(
        cols=["ratings"], separator="|", normalizer="none", imputer="none"
    )

    transformed_df = dist_multi_numerical_transormation.apply(input_df)

    transformed_rows = transformed_df.collect()

    for i, row in zip(range(1, 10, 2), transformed_rows):
        assert row["ratings"][0] == float(i)
        assert row["ratings"][1] == float(i + 1)


@pytest.mark.parametrize("delimiter", ["\\", "+", "^", "."])
def test_multi_numerical_transformation_with_special_delimiter(
    spark: SparkSession, delimiter: str, check_df_schema
):
    """Test multi-num with various special delimiters"""
    data = [
        (f"1{delimiter}2",),
        (f"3{delimiter}4",),
        (f"5{delimiter}6",),
        (f"7{delimiter}8",),
        (f"9{delimiter}10",),
    ]

    schema = StructType([StructField("ratings", StringType(), True)])
    df = spark.createDataFrame(data, schema=schema)

    dist_multi_numerical_transormation = DistMultiNumericalTransformation(
        cols=["ratings"], separator=delimiter, normalizer="none", imputer="mean"
    )

    transformed_df = dist_multi_numerical_transormation.apply(df)

    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()

    for i, row in zip(range(1, 10, 2), transformed_rows):
        assert row["ratings"][0] == float(i)
        assert row["ratings"][1] == float(i + 1)


def test_multi_numerical_transformation_with_mean_imputer(
    multi_num_df_with_missing: DataFrame, check_df_schema
):
    """Test multi-num with mean imputation"""
    dist_multi_numerical_transormation = DistMultiNumericalTransformation(
        cols=["ratings"], separator="|", normalizer="none", imputer="mean"
    )

    transformed_df = dist_multi_numerical_transormation.apply(multi_num_df_with_missing)

    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()

    for row in transformed_rows:
        # One missing val for row [3, None] with expected val (10+2+3+2)/4
        if row["ratings"][0] == 3.0:
            assert row["ratings"][1] == 4.25


def test_multi_numerical_transformation_with_minmax_scaler(
    multi_num_df_without_missing: DataFrame, check_df_schema
):
    """Test multi-num with min-max normalizer"""
    dist_multi_numerical_transormation = DistMultiNumericalTransformation(
        cols=["ratings"], separator="|", normalizer="min-max", imputer="none"
    )

    transformed_df = dist_multi_numerical_transormation.apply(multi_num_df_without_missing)

    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()

    expected_vals = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.0], [0.0, 0.125], [1.0, 0.0]]
    assert len(expected_vals) == len(transformed_rows)
    for row, expected_vector in zip(transformed_rows, expected_vals):
        assert_array_equal(row["ratings"], expected_vector)


def test_multi_numerical_transformation_with_standard_scaler(
    multi_num_df_without_missing: DataFrame,
):
    """Test multi-num with min-max normalizer"""
    dist_multi_numerical_transormation = DistMultiNumericalTransformation(
        cols=["ratings"], separator="|", normalizer="standard", imputer="none"
    )

    transformed_df = dist_multi_numerical_transormation.apply(multi_num_df_without_missing)

    # Expected: arr/arr.sum(axis=0)
    expected_vals = [
        [0.06666667, 0.52631579],
        [0.33333333, 0.10526316],
        [0.2, 0.10526316],
        [0.06666667, 0.15789474],
        [0.33333333, 0.10526316],
    ]

    transformed_rows = transformed_df.collect()
    assert len(expected_vals) == len(transformed_rows)
    for row, expected_vector in zip(transformed_rows, expected_vals):
        assert_array_almost_equal(row["ratings"], expected_vector, decimal=3)


def test_multi_numerical_transformation_with_long_vectors(spark: SparkSession, check_df_schema):
    """Test multi-num with long string vectors (768 dim)"""
    data_path = os.path.join(_ROOT, "resources/multi_num_numerical/multi_num.csv")
    long_vector_df = spark.read.csv(data_path, sep=",", header=True)
    dist_multi_numerical_transormation = DistMultiNumericalTransformation(
        cols=["feat"], separator=";", normalizer="standard", imputer="mean"
    )

    pd_df = pd.read_csv(data_path, delimiter=",")
    # Select feature vector col and convert to Series
    col_df = pd_df["feat"].apply(lambda x: [float(val) for val in x.split(";")]).apply(pd.Series)
    arr = col_df.to_numpy()
    # Divide by sum of col to get standardized vals
    expected_vals = arr / arr.sum(axis=0)

    transformed_df = dist_multi_numerical_transormation.apply(long_vector_df)
    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()
    for row, expected_vector in zip(transformed_rows, expected_vals):
        assert_array_almost_equal(row["feat"], expected_vector, decimal=3)


# TODO: Add tests for imputer requested but no NaNs


def test_multi_numerical_transformation_with_array_input(spark: SparkSession, check_df_schema):
    """Test multi-num with long array vectors (768 dim)"""
    feat_col = "feat"
    data_path = os.path.join(_ROOT, "resources/multi_num_numerical/multi_num.csv")
    # Convert the string list to an array of floats before transforming
    long_vector_df = spark.read.csv(
        data_path, sep=",", schema="id STRING, feat STRING", header=True
    )
    long_vector_df = long_vector_df.select(
        F.split(F.col(feat_col), ";").cast(ArrayType(FloatType(), True)).alias(feat_col)
    )
    dist_multi_numerical_transormation = DistMultiNumericalTransformation(
        cols=["feat"], separator=None, normalizer="standard", imputer="mean"
    )

    pd_df = pd.read_csv(data_path)
    # Select feature vector col and convert to Series
    col_df = pd_df["feat"].apply(lambda x: [float(val) for val in x.split(";")]).apply(pd.Series)
    arr = col_df.to_numpy()
    # Divide by sum of col to get standardized vals
    expected_vals = arr / arr.sum(axis=0)

    transformed_df = dist_multi_numerical_transormation.apply(long_vector_df)
    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()
    for row, expected_vector in zip(transformed_rows, expected_vals):
        assert_array_almost_equal(row["feat"], expected_vector, decimal=3)


def test_multinum_transform_default_args_vectors(
    multi_num_df_with_vectors: DataFrame, check_df_schema
):
    """Test multi-num transformation, with default arguments, and vector DF input."""
    col_name = "ratings"
    dist_multi_numerical_transormation = DistMultiNumericalTransformation(cols=[col_name])

    transformed_df = dist_multi_numerical_transormation.apply(multi_num_df_with_vectors)

    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()

    for i, row in zip(range(1, 10, 2), transformed_rows):
        assert row[col_name][0] == float(i)
        assert row[col_name][1] == float(i + 1)


def test_singlenum_transform_default_args_vectors(
    multi_num_df_with_vectors: DataFrame, check_df_schema
):
    """Test single-num transformation, with default arguments, and vector DF input."""

    col_name = "ratings"
    dist_multi_numerical_transormation = DistNumericalTransformation(cols=[col_name])

    multi_num_df_single_val = multi_num_df_with_vectors.select(
        F.slice(col_name, 1, 1).alias(col_name)
    )

    transformed_df = dist_multi_numerical_transormation.apply(multi_num_df_single_val)

    check_df_schema(transformed_df)

    transformed_rows = transformed_df.collect()

    for i, row in zip(range(1, 10, 2), transformed_rows):
        assert row[col_name][0] == float(i)


def rank_gauss(feat, eps):
    """RankGauss implementation for testing"""
    lower = -1 + eps
    upper = 1 - eps
    value_range = upper - lower
    i = np.argsort(feat, axis=0)
    j = np.argsort(i, axis=0)
    j_range = len(j) - 1
    divider = j_range / value_range
    feat = j / divider
    feat = feat - upper
    return erfinv(feat)


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
def test_rank_gauss(spark: SparkSession, check_df_schema, epsilon):
    """Test rank-Gauss transformation for value correctness."""
    data = [(0.0,), (15.0,), (26.0,), (40.0,)]

    input_df = spark.createDataFrame(data, schema=["age"])
    rg_transformation = DistNumericalTransformation(
        ["age"], imputer="none", normalizer="rank-gauss", epsilon=epsilon
    )

    output_df = rg_transformation.apply(input_df)
    check_df_schema(output_df)

    out_rows = output_df.collect()

    expected_vals = rank_gauss(np.array([[0.0], [15.0], [26.0], [40.0]]), epsilon)
    for i, row in enumerate(out_rows):
        assert_almost_equal(
            [row["age"]], expected_vals[i, :], decimal=4, err_msg=f"Row {i} is not equal"
        )


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
def test_rank_gauss_reshuffling(spark: SparkSession, check_df_schema, epsilon):
    """Test rank-Gauss transformation results order with enforced shuffling."""
    # Create DF with 10k values
    random_values = np.random.rand(10**3, 1)

    # Convert the array of values into a list of single-value tuples
    data = [(float(value),) for value in random_values]
    input_df = spark.createDataFrame(data, schema=["rand"])
    # repartition dataset
    input_df = input_df.repartition(4)
    # collect partitioned data pre-transform
    part_rows = [[row["rand"]] for row in input_df.collect()]

    rg_transformation = DistNumericalTransformation(
        ["rand"], imputer="none", normalizer="rank-gauss", epsilon=epsilon
    )

    output_df = rg_transformation.apply(input_df)
    check_df_schema(output_df)

    out_rows = output_df.collect()

    expected_vals = rank_gauss(np.array(part_rows), epsilon)
    for i, row in enumerate(out_rows):
        assert_almost_equal(
            [row["rand"]], expected_vals[i, :], decimal=4, err_msg=f"Row {i} is not equal"
        )
