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

import json
from typing import Dict, Sequence

from pyspark.sql import DataFrame, functions as F, SparkSession
from pyspark.ml.feature import StringIndexer

from .base_dist_transformation import DistributedTransformation
from . import DistMultiCategoryTransformation
from ..spark_utils import safe_rename_column


class DistSingleLabelTransformation(DistributedTransformation):
    """Transformation for single-label  classification.

    Parameters
    ----------
    cols : Sequence[str]
        A sequence of column names to be transformed. Expected to be
        a single column in the DataFrame that contains the label data.
    spark: SparkSession
        A SparkSession object, we use it to create intermediate DataFrames.
    """

    def __init__(self, cols: Sequence[str], spark: SparkSession) -> None:
        super().__init__(cols)
        self.label_column = cols[0]
        self.value_map = {}  # type: Dict[str, int]
        self.spark = spark

    def apply(self, input_df: DataFrame) -> DataFrame:
        assert self.spark
        processed_col_name = self.label_column + "_processed"

        str_indexer = StringIndexer(
            inputCol=self.label_column,
            outputCol=processed_col_name,
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )

        str_indexer_model = str_indexer.fit(input_df)
        indexed_df, self.label_column = safe_rename_column(
            str_indexer_model.transform(input_df).drop(self.label_column),
            processed_col_name,
            self.label_column,
        )

        # Labels that were missing and were assigned the value numLabels by the StringIndexer
        # are converted to None
        long_class_label = indexed_df.select(F.col(self.label_column).cast("long")).select(
            F.when(
                F.col(self.label_column) == len(str_indexer_model.labelsArray[0]),  # type: ignore
                F.lit(None),
            )
            .otherwise(F.col(self.label_column))
            .alias(self.label_column)
        )

        # Get a mapping from original label to encoded value
        label_df = self.spark.createDataFrame(
            list(enumerate(str_indexer_model.labelsArray[0])),  # type: ignore
            f"idx: int, {self.label_column}: string",
        )
        label_mapping = str_indexer_model.transform(label_df).select(
            [F.col(self.label_column), F.col(processed_col_name).cast("long")]
        )
        mapping_str_list = label_mapping.toJSON().collect()
        for mapping_str in mapping_str_list:
            map_dict = json.loads(mapping_str)
            self.value_map[map_dict[self.label_column]] = map_dict[processed_col_name]

        return long_class_label

    @staticmethod
    def get_transformation_name() -> str:
        return "DistSingleLabelTransformation"


class DistMultiLabelTransformation(DistMultiCategoryTransformation):
    """Transformation for multi-label classification.

    The label column is assumed to be a string with multiple values
    separated by the `separator`.

    Parameters
    ----------
    cols : Sequence[str]
        A sequence of column names to be transformed. Expected to be
        a single column in the DataFrame that contains the label data.
    separator: str
        The separator used to split the multi-category column into individual
        values.
    """

    def __init__(self, cols: Sequence[str], separator: str) -> None:
        super().__init__(cols, separator)
        self.label_column = cols[0]

    def apply(self, input_df: DataFrame) -> DataFrame:
        multi_cat_df = super().apply(input_df)

        return multi_cat_df
