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

from typing import Dict, List, Optional, Sequence

import numpy as np

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.functions import when
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors

from graphstorm_processing.constants import (
    MAX_CATEGORIES_PER_FEATURE,
    MISSING_CATEGORY,
    RARE_CATEGORY,
    SINGLE_CATEGORY_COL,
    SPECIAL_CHARACTERS,
)
from .base_dist_transformation import DistributedTransformation


class DistCategoryTransformation(DistributedTransformation):
    """
    Transforms categorical features into a vector of one-hot-encoded values.
    """

    def __init__(self, cols: List[str]) -> None:
        super().__init__(cols)

    @staticmethod
    def get_transformation_name() -> str:
        return "DistCategoryTransformation"

    def apply(self, input_df: DataFrame) -> DataFrame:
        processed_col_names = []
        for col in self.cols:
            processed_col_names.append(col + "_processed")
            distinct_category_counts = input_df.groupBy(col).count()  # type: DataFrame
            num_distinct_categories = distinct_category_counts.count()

            # Conditionally replace rare categories with single placeholder
            if num_distinct_categories > MAX_CATEGORIES_PER_FEATURE:
                top_categories = distinct_category_counts.orderBy("count", ascending=False).take(
                    MAX_CATEGORIES_PER_FEATURE - 1
                )
                top_categories_set = {row[0] for row in top_categories}
                # TODO: Ideally we don't want to use withColumn in a loop
                input_df = input_df.withColumn(
                    col,
                    when(input_df[col].isin(top_categories_set), input_df[col]).otherwise(
                        RARE_CATEGORY
                    ),
                )

            # Replace empty string cols with None
            input_df = input_df.withColumn(
                col, when(input_df[col] == "", None).otherwise(input_df[col])
            )

        # We first convert the strings to float indexes
        str_indexer = StringIndexer(
            inputCols=self.cols, outputCols=processed_col_names, handleInvalid="keep"
        )

        str_indexer_model = str_indexer.fit(input_df)
        indexed_df = str_indexer_model.transform(input_df)
        # Drop original columns
        indexed_df = indexed_df.select(processed_col_names)

        # Then one-hot-encode the values, produces SparseVector outputs
        sparse_vector_col_names = [col_name + "_sparse" for col_name in processed_col_names]
        one_hot_encoder = OneHotEncoder(
            inputCols=processed_col_names, outputCols=sparse_vector_col_names, handleInvalid="error"
        )

        one_hot_encoder_model = one_hot_encoder.fit(indexed_df)

        sparse_vector_features = one_hot_encoder_model.transform(indexed_df).select(
            sparse_vector_col_names
        )

        # Convert sparse feature vectors to dense
        dense_vector_features = sparse_vector_features.select(
            *[
                (vector_to_array(F.col(sparse_vector_col_name), dtype="float32")).alias(
                    orig_col_name
                )  # type: ignore
                for orig_col_name, sparse_vector_col_name in zip(self.cols, sparse_vector_col_names)
            ]
        )

        return dense_vector_features


class DistMultiCategoryTransformation(DistributedTransformation):
    """
    Transforms a multi-category column to a multi-hot representation
    of the categories. Missing values are replaced with the special
    all-zeroes vector.

    Parameters
    ----------
    cols: Sequence[str]
        List of columns to transform. Currently only supports a single column.
    separator: str
        The separator used to split the multi-category column into individual
        categories.
    """

    def __init__(self, cols: Sequence[str], separator: str) -> None:
        # TODO: Will need to have different handling for Parquet files that have vector columns
        assert len(cols) == 1, "DistMultiCategoryTransformation only supports one column at a time."
        super().__init__(cols)
        self.multi_column = cols[0]

        self.separator = separator
        # Spark's split function uses a regexp so we need to escape
        # special chars to be used as separators
        if self.separator in SPECIAL_CHARACTERS:
            self.separator = f"\\{self.separator}"

        self.value_map = {}  # type: Dict[str, int]

    @staticmethod
    def get_transformation_name() -> str:
        return "DistMultiCategoryTransformation"

    def apply(self, input_df: DataFrame) -> DataFrame:
        col_datatype = input_df.schema[self.multi_column].dataType
        is_array_col = False
        if col_datatype.typeName() == "array":
            assert isinstance(col_datatype, ArrayType)
            if not isinstance(col_datatype.elementType, StringType):
                raise ValueError(
                    f"Unsupported array type {col_datatype.elementType} "
                    f"for column {self.multi_column}, expected StringType"
                )

            is_array_col = True

        if is_array_col:
            list_df = input_df.select(self.multi_column).alias(self.multi_column)
        else:
            list_df = input_df.select(
                F.split(F.col(self.multi_column), self.separator).alias(self.multi_column)
            )

        distinct_category_counts = (
            list_df.withColumn(SINGLE_CATEGORY_COL, F.explode(F.col(self.multi_column)))
            .groupBy(SINGLE_CATEGORY_COL)
            .count()
        )

        num_distinct_categories = distinct_category_counts.count()

        # Conditionally replace rare categories with single placeholder
        if num_distinct_categories > MAX_CATEGORIES_PER_FEATURE:
            top_categories = distinct_category_counts.orderBy("count", ascending=False).take(
                MAX_CATEGORIES_PER_FEATURE - 1
            )
            top_categories_set = {row[SINGLE_CATEGORY_COL] for row in top_categories}

            # Replace rare categories with single placeholder and perform count-distinct again
            distinct_category_counts = (
                distinct_category_counts.drop("count")
                .withColumn(
                    SINGLE_CATEGORY_COL,
                    when(
                        distinct_category_counts[SINGLE_CATEGORY_COL].isin(top_categories_set),
                        distinct_category_counts[SINGLE_CATEGORY_COL],
                    ).otherwise(RARE_CATEGORY),
                )
                .groupBy(SINGLE_CATEGORY_COL)
                .count()
            )

        # Replace empty string cols with missing value placeholder
        distinct_category_counts = distinct_category_counts.withColumn(
            SINGLE_CATEGORY_COL,
            when(distinct_category_counts[SINGLE_CATEGORY_COL] == "", MISSING_CATEGORY).otherwise(
                distinct_category_counts[SINGLE_CATEGORY_COL]
            ),
        )

        # Create mapping from token to one-hot encoded vector, ordered by frequency
        ordered_categories_list = sorted(
            distinct_category_counts.collect(), key=lambda x: x["count"], reverse=True
        )
        # Remove MISSING_CATEGORY from our map
        valid_categories = [
            category[SINGLE_CATEGORY_COL]
            for category in ordered_categories_list
            if category[SINGLE_CATEGORY_COL] != MISSING_CATEGORY
        ]

        category_map = {
            category: np.array([0] * len(valid_categories)) for category in valid_categories
        }

        for i, category in enumerate(category_map.keys()):
            category_map[category][i] = 1

        # Create value map (we don't include the missing category in the map)
        self.value_map = {cat: val.nonzero()[0][0] for cat, val in category_map.items()}

        # The encoding for the missing category is an all-zeroes vector
        category_map[MISSING_CATEGORY] = np.array([0] * len(valid_categories))

        # Use mapping to convert token list to a multi-hot vector by summing one-hot vectors
        missing_vector = (
            category_map[RARE_CATEGORY]
            if num_distinct_categories > MAX_CATEGORIES_PER_FEATURE
            else category_map[MISSING_CATEGORY]
        )

        def token_list_to_multihot(token_list: Optional[List[str]]) -> Optional[List[float]]:
            if token_list:
                if len(token_list) == 1 and token_list[0] in {"NaN", "None", "null", ""}:
                    return None
                # When over MAX_CATEGORIES_PER_FEATURE we can end up with some values > 1,
                # so we threshold
                multi_hot_vec = np.where(
                    np.sum(
                        [category_map.get(token, missing_vector) for token in token_list], axis=0
                    )
                    > 0,
                    1,  # We set location to one when sum of one-hot vectors is > 0 for location
                    0,  # Zero otherwise
                )
                return Vectors.dense(multi_hot_vec).toArray().tolist()

            # When used to parse multi-labels token_list could be None/null
            return None

        token_list_to_multihot_udf = F.udf(
            token_list_to_multihot, ArrayType(FloatType(), containsNull=False)
        )

        multihot_df = list_df.withColumn(
            self.multi_column, token_list_to_multihot_udf(F.col(self.multi_column))
        )

        return multihot_df
