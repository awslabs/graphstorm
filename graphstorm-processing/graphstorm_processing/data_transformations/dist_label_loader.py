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

from dataclasses import dataclass
from math import fsum
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import NumericType

from graphstorm_processing.config.label_config_base import LabelConfig
from graphstorm_processing.data_transformations.dist_transformations import (
    DistMultiLabelTransformation,
    DistSingleLabelTransformation,
)


@dataclass
class SplitRates:
    """
    Dataclass to hold the split rates for each of the train/val/test splits.
    """

    train_rate: float
    val_rate: float
    test_rate: float

    def tolist(self) -> list[float]:
        """
        Return the split rates as a list of floats: [train_rate, val_rate, test_rate]
        """
        return [self.train_rate, self.val_rate, self.test_rate]

    def todict(self) -> dict[str, float]:
        """
        Return the split rates as a dict of str to float:
        {
            "train": train_rate,
            "val": val_rate,
            "test": test_rate,
        }
        """
        return {"train": self.train_rate, "val": self.val_rate, "test": self.test_rate}

    def __post_init__(self) -> None:
        """
        Validate the split rates.
        """
        # TODO: add support for sums <= 1.0, useful for large-scale link prediction
        if fsum([self.train_rate, self.val_rate, self.test_rate]) != 1.0:
            raise ValueError(
                "Sum of split rates must be 1.0, got "
                f"{self.train_rate=}, {self.val_rate=}, {self.test_rate=}"
            )


@dataclass
class CustomSplit:
    """
    Dataclass to hold the custom split for each of the train/val/test splits.

    Parameters
    ----------
    train : list[str]
        Paths of the training mask parquet files.
    valid : list[str]
        Paths of the validation mask parquet files.
    test : list[str]
        Paths of the testing mask parquet files.
    mask_columns : list[str]
        List of columns that contain original string ids.
    """

    train: list[str]
    valid: list[str]
    test: list[str]
    mask_columns: list[str]


class DistLabelLoader:
    """Used to transform label columns to conform to downstream GraphStorm expectations.

    Parameters
    ----------
    label_config : LabelConfig
        A configuration object that describes the label.
    spark : SparkSession
        The SparkSession to use for processing.
    """

    def __init__(
        self, label_config: LabelConfig, spark: SparkSession, order_col: Optional[str] = None
    ) -> None:
        self.label_config = label_config
        self.label_column = label_config.label_column
        self.spark = spark
        self.label_map: dict[str, int] = {}
        self.order_col = order_col

    def __str__(self) -> str:
        """Informal object representation for readability"""
        return (
            f"DistLabelLoader(label_column='{self.label_column}', "
            f"task_type='{self.label_config.task_type}', "
            f"multilabel={self.label_config.multilabel}, "
            f"order_col={self.order_col!r})"
        )

    def __repr__(self) -> str:
        """Formal object representation for debugging"""
        return (
            f"DistLabelLoader("
            f"label_config={self.label_config!r}, "
            f"spark={self.spark!r}, "
            f"order_col={self.order_col!r}, "
            f"label_map={self.label_map!r})"
        )

    def process_label(self, input_df: DataFrame) -> DataFrame:
        """Transforms the label column in the input DataFrame to conform to GraphStorm expectations.

        For single-label classification converts the input (String) column to a scalar (long).

        For multi-label classification converts the input (String) to a multi-hot binary vector.

        For regression the label column is unchanged, provided that it's a float.

        Parameters
        ----------
        input_df : DataFrame
            A Spark DataFrame that contains a label column that we will transform.

        Returns
        -------
        DataFrame
            A Spark DataFrame with the column label transformed.

        Raises
        ------
        RuntimeError
            If the label_config.task_type is not one of the supported task types,
            or if a passed in regression column is not of FloatType.
        """
        label_type = input_df.schema[self.label_column].dataType
        input_df.show()
        if self.label_config.task_type == "classification":
            assert self.order_col, f"{self.order_col} must be provided for classification tasks"
            if self.label_config.multilabel:
                assert self.label_config.separator
                label_transformer = DistMultiLabelTransformation(
                    [self.label_config.label_column], self.label_config.separator
                )
            else:
                label_transformer = DistSingleLabelTransformation(
                    [self.label_config.label_column],
                    self.spark,
                )

            transformed_label = label_transformer.apply(input_df)
            if self.order_col:
                if isinstance(self.order_col, str):
                    assert self.order_col in transformed_label.columns, (
                        f"Order column '{order_col}' not found in label dataframe, "
                        f"{transformed_label.columns=}"
                    )
                elif isinstance(self.order_col, list):
                    missing_cols = [
                        col for col in self.order_col if col not in transformed_label.columns
                    ]
                    assert not missing_cols, (
                        f"Some columns in {self.order_col=} are missing from transformed "
                        f"label DF, missing columns: {missing_cols}, got {transformed_label.columns=}"
                    )
                transformed_label = transformed_label.sort(self.order_col).cache()

            self.label_map = label_transformer.value_map
            return transformed_label
        elif self.label_config.task_type == "regression":
            if not isinstance(label_type, NumericType):
                raise RuntimeError(
                    "Data type for regression should be a NumericType, "
                    f"got {label_type} for {self.label_column}"
                )
            return input_df.select(self.label_column)
        else:
            raise RuntimeError(
                f"Unknown label task type {self.label_config.task_type} "
                f"for type: {self.label_column}"
            )
