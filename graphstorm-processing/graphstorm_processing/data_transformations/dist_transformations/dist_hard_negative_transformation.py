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
from pyspark.sql.functions import split, col
from pyspark.sql.types import ArrayType, IntegerType, StringType
from pyspark.sql import DataFrame, functions as F, SparkSession

from graphstorm_processing.constants import (
    NODE_MAPPING_STR,
    NODE_MAPPING_INT,
    ORDER_INDEX,
    EXPLODE_HARD_NEGATIVE_VALUE,
)

from .base_dist_transformation import DistributedTransformation


def apply_transform(
    cols: Sequence[str],
    separator: str,
    spark: SparkSession,
    input_df: DataFrame,
    hard_node_mapping_dict: dict,
) -> DataFrame:
    """Applies hard negative transformation to each row.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    separator: str, optional
        The separator for string input value. Only required when input value type is string.
    spark: SparkSession
        The spark session.
    input_df : DataFrame
        The input DataFrame to apply transformation to.
    hard_node_mapping_dict: dict
        The mapping dictionary contain mapping file directory and edge type.
    """
    column_type = input_df.schema[cols[0]].dataType
    if isinstance(column_type, StringType):
        transformed_df = input_df.withColumn(cols[0], split(col(cols[0]), separator))
    else:
        transformed_df = input_df
    # Edge type should be (src_ntype:relation_type:dst_ntype)
    # Only support hard negative for destination nodes. Get the node type of destination nodes.
    # TODO: support hard negative for source nodes.
    _, _, dst_type = hard_node_mapping_dict["edge_type"].split(":")
    mapping_prefix = hard_node_mapping_dict["mapping_path"]
    format_name = hard_node_mapping_dict["format_name"]
    hard_negative_node_mapping = spark.read.parquet(
        f"{mapping_prefix}{dst_type}/{format_name}/*.parquet"
    )
    node_mapping_length = hard_negative_node_mapping.count()

    # TODO: Use panda series to possibly improve the efficiency
    transformed_df = transformed_df.withColumn(ORDER_INDEX, F.monotonically_increasing_id())
    transformed_df = transformed_df.withColumn(
        EXPLODE_HARD_NEGATIVE_VALUE, F.explode(F.col(cols[0]))
    )
    transformed_df = transformed_df.join(
        hard_negative_node_mapping,
        transformed_df[EXPLODE_HARD_NEGATIVE_VALUE] == hard_negative_node_mapping[NODE_MAPPING_STR],
        "inner",
    ).select(NODE_MAPPING_INT, ORDER_INDEX)
    transformed_df = transformed_df.groupBy(ORDER_INDEX).agg(
        F.collect_list(NODE_MAPPING_INT).alias(cols[0])
    )

    # Same length for feature to convert to tensor
    def pad_mapped_values(hard_neg_list):
        while len(hard_neg_list) < node_mapping_length:
            hard_neg_list.append(-1)
        return hard_neg_list

    pad_value_udf = F.udf(pad_mapped_values, ArrayType(IntegerType()))
    transformed_df = transformed_df.orderBy(ORDER_INDEX)
    transformed_df = transformed_df.select(pad_value_udf(F.col(cols[0])).alias(cols[0]))

    return transformed_df


class DistHardEdgeNegativeTransformation(DistributedTransformation):
    """Transformation to apply hard negative transformation.

    Parameters
    ----------
    separator: str, optional
        The separator for string input value. Only required when input value type is string.
    spark: SparkSession
        The spark session.
    hard_node_mapping_dict: dict
        The mapping dictionary contain mapping file directory and edge type.
    """

    def __init__(
        self,
        cols: Sequence[str],
        spark: SparkSession,
        separator: str = "",
        hard_node_mapping_dict=None,
    ) -> None:
        super().__init__(cols, spark)
        self.cols = cols
        assert len(self.cols) == 1, "Hard Negative Transformation only supports single column"
        self.separator = separator
        self.hard_node_mapping_dict = hard_node_mapping_dict
        assert self.hard_node_mapping_dict, "edge mapping dict cannot be None for hard negative "

    def apply(self, input_df: DataFrame) -> DataFrame:
        assert self.spark
        transformed_df = apply_transform(
            self.cols, self.separator, self.spark, input_df, self.hard_node_mapping_dict
        )

        return transformed_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistHardEdgeNegativeTransformation"
