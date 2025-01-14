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


class DistHardEdgeNegativeTransformation(DistributedTransformation):
    """Transformation to apply hard negative transformation.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply hard negative transformation to.
    spark: SparkSession
        The spark session.
    hard_node_mapping_dict: dict
        The mapping dictionary contain mapping file directory and edge type.
        {
            "edge_type": str
                Edge type to apply hard negative transformation.
            "mapping_path": str
                Path to the raw node mapping.
            "format_name": str
                Parquet.
        }
    separator: str, optional
        The separator for string input value. Only required when input value type is string.
    """

    def __init__(
        self,
        cols: Sequence[str],
        spark: SparkSession,
        hard_node_mapping_dict: dict,
        separator: str = "",
    ) -> None:
        super().__init__(cols, spark)
        self.cols = cols
        assert len(self.cols) == 1, "Hard Negative Transformation only supports single column"
        self.separator = separator
        self.hard_node_mapping_dict = hard_node_mapping_dict
        assert self.hard_node_mapping_dict, "edge mapping dict cannot be None for hard negative "

    def apply(self, input_df: DataFrame) -> DataFrame:
        assert self.spark
        input_col = self.cols[0]
        column_type = input_df.schema[input_col].dataType
        if isinstance(column_type, StringType):
            transformed_df = input_df.withColumn(input_col, split(col(input_col), self.separator))
        else:
            transformed_df = input_df
        # Edge type should be (src_ntype:relation_type:dst_ntype)
        # Only support hard negative for destination nodes. Get the node type of destination nodes.
        # TODO: support hard negative for source nodes.
        _, _, dst_type = self.hard_node_mapping_dict["edge_type"].split(":")
        mapping_prefix = self.hard_node_mapping_dict["mapping_path"]
        format_name = self.hard_node_mapping_dict["format_name"]
        hard_negative_node_mapping = self.spark.read.parquet(
            f"{mapping_prefix}{dst_type}/{format_name}/"
        )
        # The maximum number of negatives in the input feature column
        max_size = (
            transformed_df.select(F.size(F.col(input_col)).alias(f"{input_col}_size"))
            .agg(F.max(f"{input_col}_size"))
            .collect()[0][0]
        )

        # TODO: Use panda series to possibly improve the efficiency
        # Explode the original list and join node id mapping dataframe
        transformed_df = transformed_df.withColumn(ORDER_INDEX, F.monotonically_increasing_id())
        # Could result in extremely large DFs in num_nodes * avg(len_of_negatives) rows
        transformed_df = transformed_df.withColumn(
            EXPLODE_HARD_NEGATIVE_VALUE, F.explode(F.col(input_col))
        )
        transformed_df = transformed_df.join(
            hard_negative_node_mapping,
            transformed_df[EXPLODE_HARD_NEGATIVE_VALUE]
            == hard_negative_node_mapping[NODE_MAPPING_STR],
            "inner",
        ).select(NODE_MAPPING_INT, ORDER_INDEX)
        transformed_df = transformed_df.groupBy(ORDER_INDEX).agg(
            F.collect_list(NODE_MAPPING_INT).alias(input_col)
        )

        # Extend the feature to the same length as the maximum length of the feature column
        def pad_mapped_values(hard_neg_list):
            if len(hard_neg_list) < max_size:
                hard_neg_list.extend([-1] * (max_size - len(hard_neg_list)))
            return hard_neg_list

        pad_value_udf = F.udf(pad_mapped_values, ArrayType(IntegerType()))
        # Make sure it keeps the original order
        transformed_df = transformed_df.orderBy(ORDER_INDEX)
        transformed_df = transformed_df.select(pad_value_udf(F.col(input_col)).alias(input_col))

        return transformed_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistHardEdgeNegativeTransformation"
