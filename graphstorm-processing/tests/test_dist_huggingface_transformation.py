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

import pytest
from pyspark.sql import DataFrame, SparkSession
import numpy as np
from numpy.testing import assert_array_equal
import torch as th
from transformers import AutoTokenizer, AutoConfig, AutoModel

from graphstorm_processing.data_transformations.dist_transformations import (
    DistHFTransformation,
)


def test_hf_tokenizer_example(spark: SparkSession, check_df_schema):
    # Prepare test data and DataFrame
    data = [
        ("mark", "doctor", None),
        ("john", "scientist", 10000),
        ("tara", "engineer", 20000),
        ("jen", "nurse", 10000),
    ]
    columns = ["name", "occupation", "salary"]
    input_df = spark.createDataFrame(data, schema=columns)

    # Configuration for Hugging Face tokenizer transformation
    hf_model = "bert-base-uncased"
    max_seq_length = 8

    # Initialize and apply the distributed Hugging Face tokenization transformation
    hf_tokenize = DistHFTransformation(["occupation"], "tokenize_hf", hf_model, max_seq_length)
    output_df = hf_tokenize.apply(input_df)
    assert (
        len(output_df.columns) == 3
    ), "the output for huggingface tokenize should have three columns"

    # Validate the schema of the transformed DataFrame
    for feature in ["input_ids", "attention_mask", "token_type_ids"]:
        feature_df = output_df.select(feature)
        check_df_schema(feature_df)

        # Collect the output data for comparison
        output_data = feature_df.collect()

        # Tokenize the original text data for validation
        original_text = [row[1] for row in data]
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        tokenized_data = tokenizer(
            original_text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        # Compare the Spark DataFrame output with the expected tokenizer output
        expected_output = tokenized_data[feature]
        for idx, row in enumerate(output_data):
            assert_array_equal(
                row[0], expected_output[idx], err_msg=f"Row {idx} for {feature} is not equal"
            )


def test_hf_emb_example(spark: SparkSession, check_df_schema):
    # Prepare test data and DataFrame
    data = [
        ("mark", "doctor", None),
        ("john", "scientist", 10000),
        ("tara", "engineer", 20000),
        ("jen", "nurse", 10000),
    ]
    columns = ["name", "occupation", "salary"]
    input_df = spark.createDataFrame(data, schema=columns)

    # Configuration for Hugging Face tokenizer transformation
    hf_model = "bert-base-uncased"
    max_seq_length = 8

    # Initialize and apply the distributed Hugging Face tokenization transformation
    hf_emb = DistHFTransformation(["occupation"], "embedding_hf", hf_model, max_seq_length)
    output_df = hf_emb.apply(input_df)

    check_df_schema(output_df)

    # Collect the output data for comparison
    output_data = output_df.collect()

    # Tokenize the original text data for validation
    original_text = [row[1] for row in data]
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    config = AutoConfig.from_pretrained(hf_model)
    lm_model = AutoModel.from_pretrained(hf_model, config)
    lm_model.eval()
    lm_model = lm_model.to("cpu")

    embeddings_list = []
    for text in original_text:
        outputs = tokenizer(text, return_tensors="pt")
        tokens = outputs["input_ids"]
        att_masks = outputs["attention_mask"]
        token_types = outputs.get("token_type_ids")
        with th.no_grad():
            if token_types is not None:
                outputs = lm_model(
                    tokens.to("cpu"),
                    attention_mask=att_masks.to("cpu"),
                    token_type_ids=token_types.to("cpu"),
                )
            else:
                outputs = lm_model(tokens.to("cpu"), attention_mask=att_masks.to("cpu"))
            embeddings = outputs.pooler_output.cpu().squeeze().numpy()
            embeddings_list.append(embeddings)

    # Compare the Spark DataFrame output with the expected tokenizer output
    expected_output = embeddings_list
    for idx, row in enumerate(output_data):
        np.testing.assert_almost_equal(
            row[0], expected_output[idx], decimal=3, err_msg=f"Row {idx} is not equal"
        )
