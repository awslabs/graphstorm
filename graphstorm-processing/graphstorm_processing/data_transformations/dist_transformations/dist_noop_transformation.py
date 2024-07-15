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

import warnings
from typing import Optional


from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, NumericType

from graphstorm_processing.constants import SPECIAL_CHARACTERS, DTYPE_MAP, TYPE_FLOAT32

from .base_dist_transformation import DistributedTransformation


class NoopTransformation(DistributedTransformation):
    """A no-op transformation that parses data as numerical values
    and forwards the result without any processing.

    For CSV input that contains numerical array rows that use a separator character,
    this transformation splits the values into a vector of doubles, e.g. "1|2|3"
    becomes a vector [1.0, 2.0, 3.0].

    Parameters
    ----------
    cols : list[str]
        The list of columns to parse as floats or lists of float
    separator : Optional[str], optional
        Optional separator to use to split the string, by default None
    out_dtype: str
        The output feature dtype
    truncate_dim: int
        When provided, will truncate the output float-vector feature to the specified dimension.
        This is useful when the feature is a multi-dimensional vector and we only need
        a subset of the dimensions, e.g. for Matryoshka Representation Learning embeddings.
    """

    def __init__(
        self,
        cols: list[str],
        out_dtype: str = TYPE_FLOAT32,
        separator: Optional[str] = None,
        truncate_dim: Optional[int] = None,
    ) -> None:
        super().__init__(cols)
        # TODO: Support multiple cols?

        self.out_dtype = out_dtype
        self.separator = separator
        # Spark's split function uses a regexp so we need to
        # escape special chars to be used as separators
        if self.separator in SPECIAL_CHARACTERS:
            self.separator = f"\\{self.separator}"
        self.truncate_dim = truncate_dim

    def _truncate_vector_df(self, input_df: DataFrame) -> DataFrame:
        """Truncates every vector in the input DF to the specified dimension."""
        assert self.truncate_dim is not None
        return input_df.select(
            [
                # SQL array indexes start at 1
                F.slice(F.col(column), 1, self.truncate_dim).alias(column)
                for column in self.cols
            ]
        )

    def apply(self, input_df: DataFrame) -> DataFrame:
        """
        Applies the transformation to the input DataFrame.
        The returned dataframe will only contain the columns specified during
        initialization of the transformation.
        """

        # If the incoming DataFrame has numerical [array] rows, just return it.
        col_datatype = input_df.schema[self.cols[0]].dataType
        if col_datatype.typeName() == "array":
            assert isinstance(col_datatype, ArrayType)
            if not isinstance(col_datatype.elementType, NumericType):
                raise ValueError(
                    f"Unsupported array type {col_datatype.elementType} "
                    f"for column {self.cols[0]}"
                )
            if self.truncate_dim:
                return self._truncate_vector_df(input_df)
            else:
                return input_df
        elif isinstance(col_datatype, NumericType):
            if self.truncate_dim is not None:
                warnings.warn(f"Trying use {self.truncate_dim=} on a DataFrame of scalars!")
            return input_df

        # Otherwise we'll try to convert the values from list of strings to list of Doubles
        def str_list_to_float_vec(string_list: Optional[list[str]]) -> Optional[list[float]]:
            if string_list:
                return [float(x) for x in string_list]
            return None

        strvec_to_float_vec_udf = F.udf(
            str_list_to_float_vec, ArrayType(DTYPE_MAP[self.out_dtype], containsNull=False)
        )

        if self.separator:
            # Split up string into vector of floats
            vector_df = input_df.select(
                [
                    strvec_to_float_vec_udf(F.split(F.col(column), self.separator)).alias(column)
                    for column in self.cols
                ]
            )
        else:
            vector_df = input_df.select(
                [
                    F.col(column).cast(DTYPE_MAP[self.out_dtype]).alias(column)
                    for column in self.cols
                ]
            )

        if self.truncate_dim:
            return self._truncate_vector_df(vector_df)
        else:
            return vector_df

    @staticmethod
    def get_transformation_name() -> str:
        """
        Get the name of the transformation
        """
        return "NoopTransformation"
