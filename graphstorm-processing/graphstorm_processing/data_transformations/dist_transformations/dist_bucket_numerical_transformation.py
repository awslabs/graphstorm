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

from .base_dist_transformation import DistributedTransformation
from .dist_numerical_transformation import apply_imputation


class DistBucketNumericalTransformation(DistributedTransformation):
    """Transformation to apply missing value imputation and bucket normalization
     to a numerical input.

    Parameters
    ----------
    cols : Sequence[str]
        The list of columns to apply the transformations on.
    range: List[float]
        The range of bucket lists only defining the start and end point.
    bucket_cnt: int
        The count of bucket lists used in the bucket feature transform.
    slide_window_size: float
        Interval or range within which numeric values are grouped into buckets.
    imputer : str
        The type of missing value imputation to apply to the column.
        Valid values are "mean", "median" and "most_frequent".
    """

    # pylint: disable=redefined-builtin
    def __init__(
        self,
        cols: List[str],
        range: List[float],
        bucket_cnt: int,
        slide_window_size: float = 0.0,
        imputer: str = "none",
    ) -> None:
        super().__init__(cols)
        self.cols = cols
        assert len(self.cols) == 1, "Bucket numerical transformation only supports single column"
        self.range = range
        self.bucket_count = bucket_cnt
        self.slide_window_size = slide_window_size
        # Spark uses 'mode' for the most frequent element
        self.shared_imputation = "mode" if imputer == "most_frequent" else imputer

    @staticmethod
    def get_transformation_name() -> str:
        return "DistBucketNumericalTransformation"

    def apply(self, input_df: DataFrame) -> DataFrame:
        imputed_df = apply_imputation(self.cols, self.shared_imputation, input_df)
        # TODO: Make range optional by getting min/max from data.
        min_val, max_val = self.range

        bucket_size = (max_val - min_val) / self.bucket_count
        epsilon = bucket_size / 10

        # TODO: Test if pyspark.ml.feature.Bucketizer covers our requirements and is faster
        def determine_bucket_membership(value: float) -> list[float]:
            # Create value range, value -> [value - slide/2, value + slide/2]
            high_val = value + self.slide_window_size / 2
            low_val = value - self.slide_window_size / 2

            # Early exits to avoid numpy calls
            membership_list = [0.0] * self.bucket_count
            if value >= max_val:
                membership_list[-1] = 1.0
                return membership_list
            if value <= min_val:
                membership_list[0] = 1.0
                return membership_list

            # Upper and lower threshold the value range
            if low_val < min_val:
                low_val = min_val
            elif low_val >= max_val:
                low_val = max_val - epsilon
            if high_val < min_val:
                high_val = min_val
            elif high_val >= max_val:
                high_val = max_val - epsilon

            # Determine upper and lower bucket membership
            low_val -= min_val
            high_val -= min_val
            low_idx = low_val // bucket_size
            high_idx = (high_val // bucket_size) + 1

            idx = np.arange(start=low_idx, stop=high_idx, dtype=int)
            membership_array = np.zeros(self.bucket_count, dtype=float)
            membership_array[idx] = 1.0

            return membership_array.tolist()

        # TODO: Try using a Pandas/Arrow UDF here and compare performance.
        bucket_udf = F.udf(determine_bucket_membership, ArrayType(FloatType()))

        bucketized_df = imputed_df.select(bucket_udf(F.col(self.cols[0])).alias(self.cols[0]))

        return bucketized_df
