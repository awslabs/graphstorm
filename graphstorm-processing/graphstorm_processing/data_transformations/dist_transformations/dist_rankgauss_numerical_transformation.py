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


class DistRankGaussNumericalTransformation(DistributedTransformation):
    """Transformation to apply missing value imputation and rank gauss normalization
     to a numerical input.

    Parameters
    ----------
    cols : Sequence[str]
        The list of columns to apply the transformations on.
    imputer : str
        The type of missing value imputation to apply to the column.
        Valid values are "mean", "median" and "most_frequent".
    """

    # pylint: disable=redefined-builtin
    def __init__(
        self,
        cols: List[str],
        imputer: str = "none",
    ) -> None:
        super().__init__(cols)
        self.cols = cols
        assert len(self.cols) == 1, "Bucket numerical transformation only supports single column"
        # Spark uses 'mode' for the most frequent element
        self.shared_imputation = "mode" if imputer == "most_frequent" else imputer

    @staticmethod
    def get_transformation_name() -> str:
        return "DistRankGaussNumericalTransformation"

    def apply(self, input_df: DataFrame) -> DataFrame:
        return None
