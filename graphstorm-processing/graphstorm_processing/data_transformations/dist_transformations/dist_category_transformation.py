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

from collections import defaultdict
from typing import List, Optional, Sequence
from functools import partial

import numpy as np
import pandas as pd

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, functions as F, SparkSession
from pyspark.sql.functions import when
from pyspark.sql.types import ArrayType, FloatType, StringType
from pyspark.sql.types import IntegerType

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

    def __init__(
        self, cols: list[str], spark: SparkSession, json_representation: Optional[dict] = None
    ) -> None:
        if not json_representation:
            json_representation = {}
        super().__init__(cols, spark, json_representation)

    @staticmethod
    def get_transformation_name() -> str:
        return "DistCategoryTransformation"

    def apply(self, input_df: DataFrame) -> DataFrame:
        assert self.spark
        processed_col_names = []
        top_categories_per_col: dict[str, list] = {}

        for current_col in self.cols:
            processed_col_names.append(current_col + "_processed")
            distinct_category_counts = input_df.groupBy(current_col).count()  # type: DataFrame
            num_distinct_categories = distinct_category_counts.count()

            # Conditionally replace rare categories with single placeholder
            if num_distinct_categories > MAX_CATEGORIES_PER_FEATURE:
                top_categories = distinct_category_counts.orderBy("count", ascending=False).take(
                    MAX_CATEGORIES_PER_FEATURE - 1
                )
                top_categories_set = {row[0] for row in top_categories}
                top_categories_per_col[current_col] = list(top_categories_set)
                # TODO: Ideally we don't want to use withColumn in a loop
                input_df = input_df.withColumn(
                    current_col,
                    when(
                        input_df[current_col].isin(top_categories_set), input_df[current_col]
                    ).otherwise(RARE_CATEGORY),
                )
            else:
                top_categories_per_col[current_col] = [
                    x[current_col] for x in distinct_category_counts.select(current_col).collect()
                ]

            # Replace empty string cols with None
            input_df = input_df.withColumn(
                current_col,
                when(input_df[current_col] == "", None).otherwise(input_df[current_col]),
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

        # Structure: {column_name: {category_string: index_value, ...}. ...}
        per_col_label_to_one_hot_idx: dict[str, dict[str, int]] = {}

        # To get the transformed values for each value in each col
        # we need to create a DataFrame with the top categories for the current
        # col, then fill in the rest of the values with placeholders
        # and pass the generated DF through the one-hot encoder
        for current_col, processed_col in zip(self.cols, processed_col_names):
            other_cols = [x for x in self.cols if x != current_col]
            top_str_categories_list = top_categories_per_col[current_col]
            # Spark doesn't include missing/unknown values in the vector
            # representation, just uses the all-zeroes vector for them,
            # so we remove instances of None from the list of strings to model
            if None in top_str_categories_list:
                top_str_categories_list.remove(None)

            # Each col might have different number of top categories, we need one DF per col
            num_current_col_cats = len(top_str_categories_list)
            # We don't care about values for the other cols in this iteration,
            # just fill with empty string
            placeholder_vals = [""] * num_current_col_cats
            placeholder_cols = [placeholder_vals for _ in range(len(self.cols) - 1)]
            current_col_unique_vals = [list(top_str_categories_list)]
            # We need to create a DF where all cols have num_rows == num_current_col_cats
            # and the current col needs to be the first col in the DF.
            vals_dict = dict(
                zip([current_col] + other_cols, current_col_unique_vals + placeholder_cols)
            )

            # One hot encoder expects a DF with all cols that were used to train it
            # so we use the top-MAX_CATEGORIES_PER_FEATURE values for the current col,
            # and the placeholders for the rest
            top_str_categories_df = self.spark.createDataFrame(pd.DataFrame(vals_dict))
            top_indexed_categories_df = str_indexer_model.transform(top_str_categories_df)

            # For the current col, get the one-hot index for each of its category strings
            # by passing the top-k values DF through the one-hot encoder model
            per_col_label_to_one_hot_idx[current_col] = {
                x[current_col]: int(x[processed_col])
                for x in one_hot_encoder_model.transform(top_indexed_categories_df).collect()
            }

        # see get_json_representation() docstring for structure
        self.json_representation = {
            "string_indexer_labels_arrays": str_indexer_model.labelsArray,
            "cols": self.cols,
            "per_col_label_to_one_hot_idx": per_col_label_to_one_hot_idx,
            "transformation_name": self.get_transformation_name(),
        }

        return dense_vector_features

    def apply_precomputed_transformation(self, input_df: DataFrame) -> DataFrame:

        # List of StringIndexerModel labelsArray lists, each one containing the strings
        # for one column. See docs for pyspark.ml.feature.StringIndexerModel.labelsArray
        labels_arrays: list[list[str]] = self.json_representation["string_indexer_labels_arrays"]
        # More verbose representation of the mapping from string to one hot index location,
        # for each column in the input.
        per_col_label_to_one_hot_idx: dict[str, dict[str, int]] = self.json_representation[
            "per_col_label_to_one_hot_idx"
        ]
        # The list of cols the transformation was originally applied to.
        precomputed_cols: list[str] = self.json_representation["cols"]

        # Assertions to ensure correctness of representation
        assert set(precomputed_cols) == set(self.cols), (
            f"Mismatched columns in precomputed transformation: "
            f"pre-computed cols: {sorted(precomputed_cols)}, "
            f"columns in current config: {sorted(self.cols)}, "
            f"different items: {set(precomputed_cols).symmetric_difference(set(self.cols))}"
        )
        for col_labels, col in zip(labels_arrays, precomputed_cols):
            for idx, label in enumerate(col_labels):
                assert idx == per_col_label_to_one_hot_idx[col][label], (
                    "Mismatch between Spark labelsArray and pre-computed array index "
                    f"for col {col}, string: {label}, "
                    f"{idx} != {per_col_label_to_one_hot_idx[col][label]}"
                )

        # For each column in the transformation, we create a defaultdict
        # with each unique value as keys, and the one-hot vector encoding
        # of the value as value. Values not in the dict get the all zeroes (missing)
        # vector
        # Do this for each column in the transformation and return the resulting DF

        # We need to define these outside the loop to avoid
        # https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/cell-var-from-loop.html
        def replace_col_in_row(val: str, str_to_vec: dict):
            return str_to_vec[val]

        def create_zeroes_list(vec_size: int):
            return [0] * vec_size

        transformed_df = None
        already_transformed_cols: list[str] = []
        remaining_cols = list(self.cols)

        for col_idx, current_col in enumerate(precomputed_cols):
            vector_size = len(labels_arrays[col_idx])
            # Mapping from string to one-hot vector,
            # with all-zeroes default for unknown/missing values
            string_to_vector: dict[str, list[int]] = defaultdict(
                partial(create_zeroes_list, vector_size)
            )

            string_to_one_hot_idx = per_col_label_to_one_hot_idx[current_col]

            # Populate the one-hot vectors for known strings
            for string_val, one_hot_idx in string_to_one_hot_idx.items():
                one_hot_vec = [0] * vector_size
                one_hot_vec[one_hot_idx] = 1
                string_to_vector[string_val] = one_hot_vec

            # UDF that replaces strings values with their one-hot encoding (ohe)
            replace_cur_col = partial(replace_col_in_row, str_to_vec=string_to_vector)
            replace_cur_col_udf = F.udf(replace_cur_col, ArrayType(IntegerType()))

            partial_df = transformed_df if transformed_df else input_df

            transformed_col = f"{current_col}_ohe"
            remaining_cols.remove(current_col)
            # We maintain only the already transformed cols, and the ones yet to be transformed
            transformed_df = partial_df.select(
                replace_cur_col_udf(F.col(current_col)).alias(transformed_col),
                *remaining_cols,
                *already_transformed_cols,
            ).drop(current_col)
            already_transformed_cols.append(transformed_col)

        assert transformed_df
        transformed_df = transformed_df.select(*already_transformed_cols).toDF(*self.cols)

        return transformed_df

    def get_json_representation(self) -> dict:
        """Representation of the single-category transformation for one or more columns.

        Returns
        -------
        dict
            Structure:
            string_indexer_labels_array:
                tuple[tuple[str]], outer tuple has num_cols elements,
                each inner tuple has num_cats elements, each str is a category string.
                Spark uses this to represent the one-hot index for each category, its
                position in the inner tuple is the one-hot-index position for the string.
                Categories are sorted by their frequency in the data.
            cols:
                list[str], with num_cols elements
            per_col_label_to_one_hot_idx:
                dict[str, dict[str, int]], with num_cols elements, each with num_categories elements
            transformation_name:
                str, will be 'DistCategoryTransformation'
        """
        return self.json_representation


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

        self.value_map: dict[str, int] = {}

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
