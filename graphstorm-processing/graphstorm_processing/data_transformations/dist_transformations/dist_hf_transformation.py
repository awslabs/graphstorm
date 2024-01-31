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
from typing import Sequence
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, IntegerType, StructType, StructField
from pyspark.sql.functions import udf
from transformers import AutoTokenizer

from graphstorm_processing.constants import HUGGINGFACE_TOKENIZE
from .base_dist_transformation import DistributedTransformation


def apply_transform(
    cols: Sequence[str], bert_norm: str, bert_model: str, max_seq_length: int, input_df: DataFrame
) -> DataFrame:
    """Applies a single normalizer to the imputed dataframe, individually to each of the columns
    provided in the cols argument.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    bert_norm : str
        The type of normalization to use. Valid values is "tokenize"
    bert_model : str
        The name of huggingface model.
    max_seq_length: int
        The maximal length of the tokenization results.
    input_df : DataFrame
        The input DataFrame to apply normalization to.
    """

    if bert_norm == HUGGINGFACE_TOKENIZE:
        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_model)

        # Define the schema of your return type
        schema = StructType(
            [
                StructField("input_ids", ArrayType(IntegerType())),
                StructField("attention_mask", ArrayType(IntegerType())),
                StructField("token_type_ids", ArrayType(IntegerType())),
            ]
        )

        # Define UDF
        @udf(returnType=schema)
        def tokenize(text):
            # Check if text is a string
            if not isinstance(text, str):
                raise ValueError("The input of the tokenizer has to be a string.")

            # Tokenize the text
            t = tokenizer(
                text,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="np",
            )
            token_type_ids = t.get("token_type_ids", np.zeros_like(t["input_ids"], dtype=np.int8))
            result = (
                t["input_ids"][0].tolist(),  # Convert tensor to list
                t["attention_mask"][0].astype(np.int8).tolist(),
                token_type_ids[0].astype(np.int8).tolist(),
            )

            return result

        # Apply the UDF to the DataFrame
        transformed_df = input_df.withColumn(cols[0], tokenize(input_df[cols[0]]))
        transformed_df = transformed_df.select(
            transformed_df[cols[0]].getItem("input_ids").alias("input_ids"),
            transformed_df[cols[0]].getItem("attention_mask").alias("attention_mask"),
            transformed_df[cols[0]].getItem("token_type_ids").alias("token_type_ids"),
        )
    return transformed_df


class DistHFTransformation(DistributedTransformation):
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
        assert len(self.cols) == 1, "Huggingface transformation only supports single column"
        self.bert_norm = normalizer
        self.bert_model = bert_model
        self.max_seq_length = max_seq_length

    def apply(self, input_df: DataFrame) -> DataFrame:
        transformed_df = apply_transform(
            self.cols, self.bert_norm, self.bert_model, self.max_seq_length, input_df
        )

        return transformed_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistHFTransformation"
