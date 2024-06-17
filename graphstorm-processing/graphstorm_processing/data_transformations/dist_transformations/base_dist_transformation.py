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

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from pyspark.sql import DataFrame, SparkSession


class DistributedTransformation(ABC):
    """
    Base class for all distributed transformations.

    Parameters
    ----------
    cols : Sequence[str]
        Column names to which we will apply the transformation
    spark : Optional[SparkSession], optional
        Optional SparkSession if needed by the underlying implementation, by default None
    json_representation : Optional[dict], optional
        Pre-computed transformation representation to use, by default None
    """

    def __init__(
        self,
        cols: Sequence[str],
        spark: Optional[SparkSession] = None,
        json_representation: Optional[dict] = None,
    ) -> None:
        self.cols = cols
        self.spark = spark
        self.json_representation = json_representation

    @abstractmethod
    def apply(self, input_df: DataFrame) -> DataFrame:
        """
        Applies the transformation to the input DataFrame, and returns the modified
        DataFrame.

        The returned DataFrame will only contain the columns specified during initialization.
        """

    def get_json_representation(self) -> dict:
        """Get a JSON representation of the transformation."""
        # TODO: Should we try to guarantee apply() has ran before this?
        if self.json_representation:
            return self.json_representation
        else:
            return {}

    def apply_precomputed_transformation(self, input_df: DataFrame) -> DataFrame:
        """Applies a transformation using pre-computed representation.

        Parameters
        ----------
        input_df : DataFrame
            Input DataFrame to apply the transformation to.

        Returns
        -------
        DataFrame
            The input DataFrame, modified according to the pre-computed transformation values.
        Raises
        ------
        NotImplementedError
            If the underlying transformation does not support re-applying using JSON input.
        """
        raise NotImplementedError(
            f"Pre-computed transformation not available for {self.get_transformation_name()}"
        )

    @staticmethod
    @abstractmethod
    def get_transformation_name() -> str:
        """Get the name of the transformation."""
