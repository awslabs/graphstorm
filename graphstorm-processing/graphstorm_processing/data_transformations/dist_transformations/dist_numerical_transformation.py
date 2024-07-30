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
from pyspark.sql.types import ArrayType
from pyspark.ml.feature import MinMaxScaler, Imputer, VectorAssembler, ElementwiseProduct
from pyspark.ml.linalg import DenseVector
from pyspark.ml.stat import Summarizer
from pyspark.ml import Pipeline
from pyspark.ml.functions import array_to_vector, vector_to_array

import numpy as np
import pandas as pd

# pylint: disable = no-name-in-module
from scipy.special import erfinv

from graphstorm_processing.constants import (
    SPECIAL_CHARACTERS,
    VALID_IMPUTERS,
    VALID_NORMALIZERS,
    DTYPE_MAP,
    TYPE_FLOAT32,
)
from .base_dist_transformation import DistributedTransformation
from ..spark_utils import rename_multiple_cols


def apply_imputation(cols: Sequence[str], shared_imputation: str, input_df: DataFrame) -> DataFrame:
    """Applies a single imputation to input dataframe, individually to each of the columns
    provided in the cols argument.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to impute.
    shared_imputation : str
        The type of imputer to use. Valid values are "none", "most_frequent"/"mode",
        "mean", "median".
    input_df : DataFrame
        The input DataFrame to apply imputations to.

    Returns
    -------
    DataFrame
        The imputed DataFrame.
    """
    # "mode" is another way to say most frequent, used by SparkML
    valid_inner_imputers = VALID_IMPUTERS + ["mode"]

    assert shared_imputation in valid_inner_imputers, (
        f"Unsupported imputation strategy requested: {shared_imputation}, the supported "
        f"strategies are : {valid_inner_imputers}"
    )
    if shared_imputation == "most_frequent":
        shared_imputation = "mode"
    if shared_imputation == "none":
        imputed_df = input_df
    else:
        imputed_col_names = [col_name + "_imputed" for col_name in cols]
        imputer = Imputer(strategy=shared_imputation, inputCols=cols, outputCols=imputed_col_names)
        model = imputer.fit(input_df)

        # Create transformed columns and drop originals, then rename transformed cols to original
        input_df = model.transform(input_df).drop(*cols)
        imputed_df, _ = rename_multiple_cols(input_df, imputed_col_names, cols)

    return imputed_df


def apply_norm(
    cols: Sequence[str],
    shared_norm: str,
    imputed_df: DataFrame,
    out_dtype: str = TYPE_FLOAT32,
    epsilon: float = 1e-6,
) -> DataFrame:
    """Applies a single normalizer to the imputed dataframe, individually to each of the columns
    provided in the cols argument.

    Parameters
    ----------
    cols : Sequence[str]
        List of column names to apply normalization to.
    shared_norm : str
        The type of normalization to use. Valid values are "none", "min-max",
        "standard", "rank-gauss".
    imputed_df : DataFrame
        The input DataFrame to apply normalization to. It should not contain
        missing values.
    out_dtype: str
        The output feature dtype.
    epsilon: float
        Epsilon for normalization used to avoid INF float during computation
        on "rank-gauss".

    Returns
    -------
    DataFrame
        The normalized DataFrame with only the columns listed in `cols` retained.

    Raises
    ------
    RuntimeError
        If missing values exist in the data when the "standard" normalizer is used.

    ValueError
        If unsupported feature output dtype is provided.
    """
    other_cols = list(set(imputed_df.columns).difference(cols))

    def single_vec_to_float(vec):
        return float(vec[0])

    # Use the map to get the corresponding data type object, or raise an error if not found
    if out_dtype in DTYPE_MAP:
        vec_udf = F.udf(single_vec_to_float, DTYPE_MAP[out_dtype])
    else:
        raise ValueError("Unsupported feature output dtype")

    assert shared_norm in VALID_NORMALIZERS, (
        f"Unsupported normalization requested: {shared_norm}, the supported "
        f"strategies are : {VALID_NORMALIZERS}"
    )

    if shared_norm == "none":
        # Save the time and efficiency for not casting the type
        # when not doing any normalization
        scaled_df = imputed_df
    elif shared_norm == "min-max":
        # Because the scalers expect Vector input, we need to use VectorAssembler on each,
        # creating one (scaled) vector per normalizer type
        # TODO: See if it's possible to have all features under one assembler and scaler,
        # speeding up the process. Then do the "disentaglement" on the caller side.
        assemblers = [VectorAssembler(inputCols=[col], outputCol=col + "_vec") for col in cols]
        scalers = [MinMaxScaler(inputCol=col + "_vec", outputCol=col + "_scaled") for col in cols]

        vector_cols = [col + "_vec" for col in cols]
        scaled_cols = [col + "_scaled" for col in cols]
        pipeline = Pipeline(stages=assemblers + scalers)
        scaler_model = pipeline.fit(imputed_df)
        scaled_df = scaler_model.transform(imputed_df).drop(*vector_cols).drop(*cols)

        scaled_df = scaled_df.select(
            *[
                (vec_udf(F.col(scaled_col_name))).alias(orig_col)
                for scaled_col_name, orig_col in zip(scaled_cols, cols)
            ]
            + other_cols
        )
    elif shared_norm == "standard":
        col_sums = imputed_df.agg({col: "sum" for col in cols}).collect()[0].asDict()
        # TODO: See if it's possible to exclude NaN values from the sum
        for _, val in col_sums.items():
            if np.isinf(val) or np.isnan(val):
                raise RuntimeError(
                    "Missing values found in the data, cannot apply "
                    "normalization. Use an imputer in the transformation."
                )
        scaled_df = imputed_df.select(
            [(F.col(c) / col_sums[f"sum({c})"]).cast(DTYPE_MAP[out_dtype]).alias(c) for c in cols]
            + other_cols
        )
    elif shared_norm == "rank-gauss":
        assert len(cols) == 1, "Rank-Guass numerical transformation only supports single column"
        column_name = cols[0]
        select_df = imputed_df.select(column_name)
        # original id is the original order for the input data frame,
        # value rank indicates the rank of each value in the column
        # We need original id to help us restore the order.
        original_order_col = f"original-order-{uuid.uuid4().hex[8]}"
        value_rank_col = f"value-rank-{uuid.uuid4().hex[8]}"
        df_with_order_idx = select_df.withColumn(
            original_order_col, F.monotonically_increasing_id()
        )
        value_sorted_df = df_with_order_idx.orderBy(column_name)
        value_rank_df = value_sorted_df.withColumn(value_rank_col, F.monotonically_increasing_id())

        # pylint: disable = cell-var-from-loop
        # It is required to put num_rows definition outside,
        # or pandas.udf will throw an error
        def gauss_transform(rank: pd.Series) -> pd.Series:
            feat_range = num_rows - 1
            clipped_rank = (rank / feat_range - 0.5) * 2
            clipped_rank = np.maximum(np.minimum(clipped_rank, 1 - epsilon), epsilon - 1)
            return pd.Series(erfinv(clipped_rank))

        num_rows = value_rank_df.count()
        gauss_udf = F.pandas_udf(gauss_transform, DTYPE_MAP[out_dtype])
        normalized_df = value_rank_df.withColumn(column_name, gauss_udf(value_rank_col))
        scaled_df = normalized_df.orderBy(original_order_col).drop(
            value_rank_col, original_order_col
        )

    return scaled_df


class DistNumericalTransformation(DistributedTransformation):
    """Transformation to apply missing value imputation and various forms of normalization
     to a numerical input.

    Parameters
    ----------
    cols : Sequence[str]
        The list of columns to apply the transformations on.
    normalizer : str
        The normalization to apply to the columns.
        Valid values are "none", "min-max", "standard", "rank-gauss".
    imputer : str
        The type of missing value imputation to apply to the column.
        Valid values are "mean", "median" and "most_frequent".
    out_dtype: str
        Output feature dtype
    epsilon: float
        Epsilon for normalization used to avoid INF float during computation.
    """

    def __init__(
        self,
        cols: Sequence[str],
        normalizer: str = "none",
        imputer: str = "none",
        out_dtype: str = TYPE_FLOAT32,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__(cols)
        self.cols = cols
        self.shared_norm = normalizer
        self.epsilon = epsilon
        self.out_dtype = out_dtype
        # Spark uses 'mode' for the most frequent element
        self.shared_imputation = "mode" if imputer == "most_frequent" else imputer

    def apply(self, input_df: DataFrame) -> DataFrame:
        logging.debug(
            "Applying normalizer: %s, imputation: %s", self.shared_norm, self.shared_imputation
        )

        imputed_df = apply_imputation(self.cols, self.shared_imputation, input_df)
        scaled_df = apply_norm(
            self.cols, self.shared_norm, imputed_df, self.out_dtype, self.epsilon
        )

        # TODO: Figure out why the transformation is producing Double values, and switch to float
        return scaled_df

    @staticmethod
    def get_transformation_name() -> str:
        return "DistNumericalTransformation"


class DistMultiNumericalTransformation(DistNumericalTransformation):
    """Transformation to apply missing value imputation and normalization
     to a multi-column numerical input.

    Parameters
    ----------
    cols : Sequence[str]
        The list of columns to apply the transformations on.
    separator: str
        The separator string that divides the string values.
    normalizer : str
        The normalization to apply to the columns.
        Valid values are "none", "min-max", and "standard".
    imputer : str
        The type of missing value imputation to apply to the column.
        Valid values are "mean", "median" and "most_frequent".
    out_dtype: str
        Output feature dtype
    """

    def __init__(
        self,
        cols: Sequence[str],
        separator: Optional[str] = None,
        normalizer: str = "none",
        imputer: str = "none",
        out_dtype: str = TYPE_FLOAT32,
    ) -> None:
        assert (
            len(cols) == 1
        ), "DistMultiNumericalTransformation only supports one column at a time."
        super().__init__(cols, normalizer, imputer)
        self.multi_column = cols[0]

        self.separator = separator
        # Keep the original separator to split in pure Python
        self.original_separator = self.separator
        # Spark's split function uses a regexp so we need to escape
        # special chars to be used as separators
        if self.separator in SPECIAL_CHARACTERS:
            self.separator = f"\\{self.separator}"
        self.out_dtype = out_dtype

    @staticmethod
    def get_transformation_name() -> str:
        return "DistMultiNumericalTransformation"

    @staticmethod
    def apply_norm_vector(vector_col: str, shared_norm: str, vector_df: DataFrame) -> DataFrame:
        """
        Applies normalizer column-wise with a single vector column as input.
        """
        other_cols = list(set(vector_df.columns).difference([vector_col]))

        if shared_norm == "none":
            scaled_df = vector_df
        elif shared_norm == "min-max":
            min_max_scaler = MinMaxScaler(inputCol=vector_col, outputCol=vector_col + "_scaled")

            scaler_model = min_max_scaler.fit(vector_df)
            scaled_df = scaler_model.transform(vector_df).drop(vector_col)

            scaled_df = scaled_df.withColumnRenamed(vector_col + "_scaled", vector_col)
            scaled_df = scaled_df.select([vector_col] + other_cols)
        elif shared_norm == "standard":
            col_sums_df = vector_df.select(Summarizer.sum(vector_df[vector_col]).alias("sum"))

            col_sums = col_sums_df.collect()[0]["sum"]  # type: DenseVector

            col_sums_array = col_sums.toArray()
            for i, col_sum in enumerate(col_sums_array):
                if np.isinf(col_sum) or np.isnan(col_sum) or col_sum == 0:
                    col_sums_array[i] = 0.0
                else:
                    col_sums_array[i] = 1.0 / col_sums_array[i]

            elwise_divider = ElementwiseProduct(
                scalingVec=DenseVector(col_sums_array),
                inputCol=vector_col,
                outputCol=f"{vector_col}_scaled",
            )

            scaled_df = elwise_divider.transform(vector_df).drop(vector_col)

            scaled_df = scaled_df.withColumnRenamed(vector_col + "_scaled", vector_col)
            scaled_df = scaled_df.select([vector_col] + other_cols)
        else:
            raise RuntimeError(f"Unknown normalizer requested for col {vector_col}: {shared_norm}")

        return scaled_df

    def apply(self, input_df: DataFrame) -> DataFrame:
        def replace_empty_with_nan(x):
            return F.when(x == "", "NaN").otherwise(x)

        def convert_multistring_to_sequence_df(
            multi_string_df: DataFrame, separator: str, column_type: str
        ) -> DataFrame:
            """
            Convert the provided DataFrame, that is assumed to have one string
            column named with the value of `self.multi_column`,
            to a single-column sequence DF with the same column name.

            If `column_type` is "array", the returned DF has one ArrayType column,
            otherwise if `column_type` is "vector" the DF has one DenseVector type column.
            """
            assert column_type in ["array", "vector"]
            # Hint: read the transformation comments inside-out, starting with split
            array_df = multi_string_df.select(
                # After split, replace empty strings with 'NaN' and cast to Array<Float>
                F.transform(
                    F.split(
                        multi_string_df[self.multi_column], separator
                    ),  # Split along the separator
                    replace_empty_with_nan,
                )
                .cast(ArrayType(DTYPE_MAP[self.out_dtype], True))
                .alias(self.multi_column)
            )

            if column_type == "array":
                return array_df

            vector_df = array_df.select(
                # Take array column and convert to a DenseVector column
                array_to_vector(array_df[self.multi_column]).alias(self.multi_column)
            )
            return vector_df

        def vector_df_has_nan(vector_df: DataFrame, vector_col: str) -> bool:
            """
            Returns true if there exists at least one NaN value in `vector_df[vector_col]`
            """
            sum_vector_df = vector_df.select(Summarizer.mean(vector_df[vector_col]).alias("sum"))
            sums_vector = sum_vector_df.take(1)[0]["sum"]
            for val in sums_vector:
                if np.isnan(val):
                    return True
            return False

        # Convert the input column from either a delimited string or array to a Vector column
        multi_col_type = input_df.schema.jsonValue()["fields"][0]["type"]
        if multi_col_type == "string":
            assert self.separator, "Separator needed when dealing with CSV multi-column data."
            vector_df = convert_multistring_to_sequence_df(
                input_df, self.separator, column_type="vector"
            )
        else:
            vector_df = input_df.select(
                array_to_vector(F.col(self.multi_column)).alias(self.multi_column)
            )

        if self.shared_imputation != "none":
            # First we check if any NaN values exist in the input DF
            # TODO: Replace column-level NaN search with row-level missing which should be faster.
            input_has_nan = vector_df_has_nan(vector_df, self.multi_column)

            if input_has_nan:
                # If the input has NaN, check how values exist in the vectors
                feat_row = input_df.take(1)[0]

                len_array = len(feat_row[self.multi_column].split(self.original_separator))

                # Doing the column splitting in memory can be expensive
                if len_array > 50:
                    logging.warning(
                        "Attempting imputation on %d columns can lead to OOM errors", len_array
                    )

                # Splitting the vectors requires an array DF
                if multi_col_type == "string":
                    assert self.separator, "Separator must be provided for string-separated vectors"
                    split_array_df = convert_multistring_to_sequence_df(
                        input_df, self.separator, column_type="array"
                    )
                else:
                    split_array_df = input_df.select(
                        F.col(self.multi_column)
                        .cast(ArrayType(DTYPE_MAP[self.out_dtype], True))
                        .alias(self.multi_column)
                    )

                # Split the values into separate columns
                split_col_df = split_array_df.select(
                    [F.col(self.multi_column)[i] for i in range(len_array)]
                )

                # Set the newly created columns as the ones to be processed,
                # and call the base numerical transformer
                imputed_df = apply_imputation(
                    split_col_df.columns, self.shared_imputation, split_col_df
                )

                # Assemble the separate columns back into a single vector column
                assembler = VectorAssembler(
                    inputCols=imputed_df.columns, outputCol=self.multi_column, handleInvalid="keep"
                )

                imputed_df = assembler.transform(imputed_df).drop(*imputed_df.columns)
            else:
                # If there input_df had no NaN values, we just pass the vector df to the scaler
                imputed_df = vector_df
        else:
            # If the user did not request any imputation, just pass to the scaler
            imputed_df = vector_df

        scaled_df = self.apply_norm_vector(self.multi_column, self.shared_norm, imputed_df)

        # Convert DenseVector data to List[float]
        scaled_df = scaled_df.select(
            vector_to_array(scaled_df[self.multi_column]).alias(self.multi_column)
        )

        return scaled_df
