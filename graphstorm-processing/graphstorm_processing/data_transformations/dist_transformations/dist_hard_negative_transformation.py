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

import logging
import os
from typing import Sequence
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf

from .base_dist_transformation import DistributedTransformation


def apply_transform(
    cols: Sequence[str], separator: str, input_df: DataFrame, edge_mapping_dict: dict
) -> DataFrame:
    """Applies hard negative transformation to each row.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    separator: str, optional
        The separator for string input value. Only required when input value type is string.
    input_df : DataFrame
        The input DataFrame to apply normalization to.
    edge_mapping_dict: dict
        The mapping dictionary contain mapping file directory and edge type
    """

    input_df.show()
    print(edge_mapping_dict)
    exit(-1)

    return transformed_df


class DistHardNegativeTransformation(DistributedTransformation):
    """Transformation to apply hard negative transformation.

    Parameters
    ----------
    separator: str, optional
        The separator for string input value. Only required when input value type is string.
    """

    def __init__(
        self, cols: Sequence[str], separator: str = "", edge_mapping_dict=None
    ) -> None:
        super().__init__(cols)
        self.cols = cols
        assert len(self.cols) == 1, "Hard Negative Transformation only supports single column"
        self.separator = separator
        self.edge_mapping_dict = edge_mapping_dict
        assert self.edge_mapping_dict, "edge mapping dict cannot be None for hard negative "

    def apply(self, input_df: DataFrame) -> DataFrame:
        transformed_df = apply_transform(
            self.cols, self.separator, input_df, self.edge_mapping_dict
        )

        return transformed_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistHardNegativeTransformation"
