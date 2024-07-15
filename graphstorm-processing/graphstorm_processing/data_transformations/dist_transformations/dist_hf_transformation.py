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
import numpy as np
import torch as th
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, IntegerType, FloatType, StructType, StructField
from pyspark.sql.functions import udf
from transformers import AutoTokenizer, AutoModel, AutoConfig

from graphstorm_processing.constants import HUGGINGFACE_TOKENIZE, HUGGINGFACE_EMB
from .base_dist_transformation import DistributedTransformation


def apply_transform(
    cols: Sequence[str], action: str, hf_model: str, max_seq_length: int, input_df: DataFrame
) -> DataFrame:
    """Applies a single normalizer to the imputed dataframe, individually to each of the columns
    provided in the cols argument.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    action : str
        The type of normalization to use. Valid values are ["tokenize_hf", "embedding_hf"].
    hf_model : str
        The name of huggingface model.
    max_seq_length: int
        The maximal length of the tokenization results.
    input_df : DataFrame
        The input DataFrame to apply normalization to.
    """

    if action == HUGGINGFACE_TOKENIZE:
        # Initialize the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        if max_seq_length > tokenizer.model_max_length:
            raise RuntimeError(
                f"max_seq_length {max_seq_length} is larger "
                f"than expected {tokenizer.model_max_length}"
            )
        # Define the schema of your return type
        tokenize_schema = StructType(
            [
                StructField("input_ids", ArrayType(IntegerType())),
                StructField("attention_mask", ArrayType(IntegerType())),
                StructField("token_type_ids", ArrayType(IntegerType())),
            ]
        )

        # Define UDF
        @udf(returnType=tokenize_schema)
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
    elif action == HUGGINGFACE_EMB:
        # Define the schema of your return type
        embedding_schema = ArrayType(FloatType())

        if th.cuda.is_available():
            gpu = (
                int(os.environ["CUDA_VISIBLE_DEVICES"])
                if "CUDA_VISIBLE_DEVICES" in os.environ
                else 0
            )
            device = f"cuda:{gpu}"
        else:
            device = "cpu"
        logging.warning("The device to run huggingface transformation is %s", device)
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        if max_seq_length > tokenizer.model_max_length:
            # TODO: Could we possibly raise this at config time?
            raise RuntimeError(
                f"max_seq_length {max_seq_length} is larger "
                f"than expected {tokenizer.model_max_length}"
            )
        config = AutoConfig.from_pretrained(hf_model)
        lm_model = AutoModel.from_pretrained(hf_model, config)
        lm_model.eval()
        lm_model = lm_model.to(device)

        # Define UDF
        @udf(returnType=embedding_schema)
        def lm_emb(text):
            # Check if text is a string
            if not isinstance(text, str):
                raise ValueError("The input of the tokenizer has to be a string.")

            # Tokenize the text
            outputs = tokenizer(
                text,
                max_length=max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            token_type_ids = outputs.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = th.zeros_like(outputs["input_ids"], dtype=th.int8)
            with th.no_grad():
                lm_outputs = lm_model(
                    input_ids=outputs["input_ids"].to(device),
                    attention_mask=outputs["attention_mask"].to(device).long(),
                    token_type_ids=token_type_ids.to(device).long(),
                )
                embeddings = lm_outputs.pooler_output.cpu().squeeze().numpy()
            return embeddings.tolist()

        # Apply the UDF to the DataFrame
        transformed_df = input_df.select(lm_emb(input_df[cols[0]]).alias(cols[0]))
    else:
        raise ValueError(f"The input action needs to be {HUGGINGFACE_TOKENIZE}")

    return transformed_df


class DistHFTransformation(DistributedTransformation):
    """Transformation to apply various forms of bert normalization to a text input.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    action : str
        The type of huggingface action to use. Valid values are ["tokenize_hf", "embedding_hf"].
    hf_model: str, required
        The name of the lm model.
    max_seq_length: int, required
        The maximal length of the tokenization results.
    """

    def __init__(
        self, cols: Sequence[str], action: str, hf_model: str, max_seq_length: int
    ) -> None:
        super().__init__(cols)
        self.cols = cols
        assert len(self.cols) == 1, "Huggingface transformation only supports single column"
        self.action = action
        self.hf_model = hf_model
        self.max_seq_length = max_seq_length

    def apply(self, input_df: DataFrame) -> DataFrame:
        transformed_df = apply_transform(
            self.cols, self.action, self.hf_model, self.max_seq_length, input_df
        )

        return transformed_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistHFTransformation"
