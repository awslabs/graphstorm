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
import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy.special import erfinv
from graphstorm_processing.data_transformations.dist_transformations import (
    DistRankGaussNumericalTransformation,
)


def rank_gauss(feat, eps):
    lower = -1 + eps
    upper = 1 - eps
    range = upper - lower
    i = np.argsort(feat, axis=0)
    j = np.argsort(i, axis=0)
    j_range = len(j) - 1
    divider = j_range / range
    feat = j / divider
    feat = feat - upper
    return erfinv(feat)


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
def test_rank_guass(spark: SparkSession, check_df_schema, epsilon):
    data = [("mark", 0.0, None), ("john", 15.0, 10000), ("tara", 26.0, 20000), ("jen", 40.0, 10000)]

    columns = ["name", "age", "salary"]
    input_df = spark.createDataFrame(data, schema=columns)
    rg_transformation = DistRankGaussNumericalTransformation(
        ["age"], epsilon=epsilon
    )

    output_df = rg_transformation.apply(input_df)
    check_df_schema(output_df)

    out_rows = output_df.collect()

    expected_vals = rank_gauss(np.array([[0.0], [15.0], [26.0], [40.0]]), epsilon)
    for i, row in enumerate(out_rows):
        assert_almost_equal([row["age"]], expected_vals[i, :], decimal=4, err_msg=f"Row {i} is not equal")
