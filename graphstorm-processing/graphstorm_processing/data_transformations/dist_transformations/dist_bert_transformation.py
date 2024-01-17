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
from pyspark.sql.types import MapType, ArrayType, IntegerType, StringType
from pyspark.ml.stat import Summarizer
from pyspark.ml import Pipeline
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import udf

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from .base_dist_transformation import DistributedTransformation
from ..spark_utils import rename_multiple_cols


def apply_norm(
    cols: Sequence[str], bert_norm: str, max_seq_length: int, input_df: DataFrame
) -> DataFrame:
    """Applies a single normalizer to the imputed dataframe, individually to each of the columns
    provided in the cols argument.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    bert_norm : str
        The type of normalization to use. Valid values is "tokenize"
    max_seq_length : int
        The maximal length of the tokenization results.
    input_df : DataFrame
        The input DataFrame to apply normalization to.
    """

    if bert_norm == "tokenize":
        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tokenize(text):
            # Check if text is a string
            if not isinstance(text, str):
                raise ValueError("The input of the tokenizer has to be a string.")

            # Tokenize the text
            t = tokenizer(text, max_length=max_seq_length, truncation=True, padding='max_length', return_tensors='np')
            token_type_ids = t.get('token_type_ids', np.zeros_like(t['input_ids'], dtype=np.int8))
            result = {
                'input_ids': t['input_ids'][0].tolist(),  # Convert tensor to list
                'attention_mask': t['attention_mask'][0].astype(np.int8).tolist(),
                'token_type_ids': token_type_ids[0].astype(np.int8).tolist()
            }
            return result

        # Define the UDF with the appropriate return type
        tokenize_udf = udf(tokenize, MapType(StringType(), ArrayType(IntegerType())))

        # Apply the UDF to the DataFrame
        scaled_df = input_df.withColumn(cols[0], tokenize_udf(input_df[cols[0]]))

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
        assert len(self.cols) == 1, "Bert transformation only supports single column"
        self.bert_norm = normalizer
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length

    def apply(self, input_df: DataFrame) -> DataFrame:
        scaled_df = apply_norm(self.cols, self.bert_norm, self.max_seq_length, input_df)

        return scaled_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistBertTransformation"