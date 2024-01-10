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
from typing import Optional, Sequence
import uuid

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import MinMaxScaler, Imputer, VectorAssembler, ElementwiseProduct
from pyspark.ml.linalg import DenseVector
from pyspark.ml.stat import Summarizer
from pyspark.ml import Pipeline
from pyspark.ml.functions import array_to_vector, vector_to_array

import numpy as np
import pandas as pd

from .base_dist_transformation import DistributedTransformation
from ..spark_utils import rename_multiple_cols


def apply_norm(
    cols: Sequence[str], bert_norm: str, input_df: DataFrame
) -> DataFrame:
    """Applies a single normalizer to the imputed dataframe, individually to each of the columns
    provided in the cols argument.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    bert_norm : str
        The type of normalization to use. Valid values is "tokenize"
    input_df : DataFrame
        The input DataFrame to apply normalization to.
    """

    if bert_norm == "tokenize":
        scaled_df = input_df

    return scaled_df


class DistBertTransformation(DistributedTransformation):
    """Transformation to apply various forms of bert normalization to a text input.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    bert_norm : str
        The type of normalization to use. Valid values is "tokenize"
    """

    def __init__(
        self, cols: Sequence[str], normalizer: str, bert_model: str, max_seq_length: int
    ) -> None:
        super().__init__(cols)
        self.cols = cols
        self.bert_norm = normalizer
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length

    def apply(self, input_df: DataFrame) -> DataFrame:
        scaled_df = apply_norm(self.cols, self.bert_norm, input_df)

        return scaled_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistBertTransformation"