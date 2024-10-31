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
from pyspark.sql.functions import udf, split, col
from pyspark.sql.types import ArrayType, IntegerType, StringType
from pyspark.sql import DataFrame, functions as F, SparkSession

from .base_dist_transformation import DistributedTransformation

from graphstorm_processing.constants import NODE_MAPPING_STR, NODE_MAPPING_INT

def apply_transform(
    cols: Sequence[str], separator: str, spark: SparkSession, input_df: DataFrame, edge_mapping_dict: dict
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
    column_type = input_df.schema[cols[0]].dataType
    if isinstance(column_type, StringType):
        transformed_df = input_df.withColumn(cols[0], split(col(cols[0]), separator))
    else:
        transformed_df = input_df
    # Edge type should be (src_ntype:get_relation_name()}:dst_ntype)
    # Assume all the node type in the hard negative feature should be dst node type
    _, _, dst_type = edge_mapping_dict["edge_type"].split(":")
    mapping_prefix = edge_mapping_dict["mapping_path"]
    format_name = edge_mapping_dict["format_name"]
    hard_negative_node_mapping = spark.read.parquet(f"{mapping_prefix}{dst_type}/{format_name}/*.parquet")
    node_mapping_length = hard_negative_node_mapping.count()

    # TODO: This method may suffer from scalability issue, we can make this method to join-based solution.
    hard_negative_node_mapping_dict = {row[NODE_MAPPING_STR]: row[NODE_MAPPING_INT] for row in hard_negative_node_mapping.collect()}

    # Same length for feature to convert to tensor
    def map_values(hard_neg_list):
        mapped_values = [hard_negative_node_mapping_dict.get(item, -1) for item in hard_neg_list]
        while len(mapped_values) < node_mapping_length:
            mapped_values.append(-1)
        return mapped_values

    map_values_udf = F.udf(map_values, ArrayType(IntegerType()))

    transformed_df = transformed_df.select(map_values_udf(F.col(cols[0])).alias(cols[0]))
    return transformed_df


class DistHardNegativeTransformation(DistributedTransformation):
    """Transformation to apply hard negative transformation.

    Parameters
    ----------
    separator: str, optional
        The separator for string input value. Only required when input value type is string.
    spark: SparkSession
        The spark session
    edge_mapping_dict: dict
        The node type and mapping directory
    """

    def __init__(
        self, cols: Sequence[str], spark: SparkSession, separator: str = "", edge_mapping_dict=None
    ) -> None:
        super().__init__(cols, spark)
        self.cols = cols
        assert len(self.cols) == 1, "Hard Negative Transformation only supports single column"
        self.separator = separator
        self.edge_mapping_dict = edge_mapping_dict
        assert self.edge_mapping_dict, "edge mapping dict cannot be None for hard negative "

    def apply(self, input_df: DataFrame) -> DataFrame:
        assert self.spark
        transformed_df = apply_transform(
            self.cols, self.separator, self.spark, input_df, self.edge_mapping_dict
        )

        return transformed_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistHardNegativeTransformation"
