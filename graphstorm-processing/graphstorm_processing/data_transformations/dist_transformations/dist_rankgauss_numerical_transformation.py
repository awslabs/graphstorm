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
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
import pandas as pd
from scipy.special import erfinv

from .base_dist_transformation import DistributedTransformation
from .dist_numerical_transformation import apply_imputation


class DistRankGaussNumericalTransformation(DistributedTransformation):
    """Transformation to apply missing value imputation and rank gauss normalization
     to a numerical input.

    Parameters
    ----------
    cols : Sequence[str]
        The list of columns to apply the transformations on.
    epsilon: float
        Epsilon for normalization used to avoid INF float during computation.
    imputer : str
        The type of missing value imputation to apply to the column.
        Valid values are "mean", "median" and "most_frequent".
    """

    def __init__(
        self,
        cols: List[str],
        epsilon: float = 0.0,
        imputer: str = "none",
    ) -> None:
        super().__init__(cols)
        self.cols = cols
        assert len(self.cols) == 1, "Rank Guass numerical transformation only supports single column"
        # Spark uses 'mode' for the most frequent element
        self.shared_imputation = "mode" if imputer == "most_frequent" else imputer
        self.epsilon = epsilon

    @staticmethod
    def get_transformation_name() -> str:
        return "DistRankGaussNumericalTransformation"

    def apply(self, input_df: DataFrame) -> DataFrame:
        column_name = self.cols[0]
        select_df = input_df.select(column_name)
        imputed_df = apply_imputation(self.cols, self.shared_imputation, select_df)

        id_df = imputed_df.withColumn('id', F.monotonically_increasing_id())
        sorted_df = id_df.orderBy(column_name)
        indexed_df = sorted_df.withColumn('index', F.monotonically_increasing_id())

        def gauss_transform(rank: pd.Series) -> pd.Series:
            epsilon = self.epsilon
            feat_range = num_rows - 1
            clipped_rank = (rank / feat_range - 0.5) * 2
            clipped_rank = np.maximum(np.minimum(clipped_rank, 1 - epsilon), epsilon - 1)
            return pd.Series(erfinv(clipped_rank))

        gauss_udf = F.pandas_udf(gauss_transform, FloatType())
        num_rows = indexed_df.count()
        normalized_df = indexed_df.withColumn(column_name, gauss_udf('index'))
        gauss_transformed_df = normalized_df.orderBy('id').drop("index", "id")

        return gauss_transformed_df
