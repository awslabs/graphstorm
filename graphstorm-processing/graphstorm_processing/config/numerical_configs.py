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
from typing import Mapping

from graphstorm_processing.constants import VALID_IMPUTERS, VALID_NORMALIZERS
from .feature_config_base import FeatureConfig


class NumericalFeatureConfig(FeatureConfig):
    """Feature configuration for single-column numerical features.

    Supported kwargs
    ----------------
    imputer: str
        A method to fill in missing values in the data. Valid values are:
        "mean" (Default), "median", and "most_frequent". Missing values will be replaced
        with the respective value computed from the data.

    normalizer: str
        A normalization to apply to each column. Valid values are
        "none", "min-max", and "standard".

        The transformation applied will be:

        * "none": (Default) Don't normalize the numerical values during encoding.
        * "min-max": Normalize each value by subtracting the minimum value from it,
        and then dividing it by the difference between the maximum value and the minimum.
        * "standard": Normalize each value by dividing it by the sum of all the values.
    """

    def __init__(self, config: Mapping):
        super().__init__(config)
        self.imputer = self._transformation_kwargs.get("imputer", "mean")
        self.norm = self._transformation_kwargs.get("normalizer", "none")

        self._sanity_check()

    def _sanity_check(self) -> None:
        super()._sanity_check()
        assert (
            self.imputer in VALID_IMPUTERS
        ), f"Unknown imputer requested, expected one of {VALID_IMPUTERS}, got {self.imputer}"
        assert (
            self.norm in VALID_NORMALIZERS
        ), f"Unknown normalizer requested, expected one of {VALID_NORMALIZERS}, got {self.norm}"


class MultiNumericalFeatureConfig(NumericalFeatureConfig):
    """Feature configuration for multi-column numerical features.

    Supported kwargs
    ----------------
    imputer: str
        A method to fill in missing values in the data. Valid values are:
        "mean" (Default), "median", and "most_frequent". Missing values will be replaced
        with the respective value computed from the data.

    normalizer: str
        A normalization to apply to each column. Valid values are
        "none", "min-max", and "standard".

        The transformation applied will be:

        * "none": (Default) Don't normalize the numerical values during encoding.
        * "min-max": Normalize each value by subtracting the minimum value from it,
        and then dividing it by the difference between the maximum value and the minimum.
        * "standard": Normalize each value by dividing it by the sum of all the values.

    separator: str, optional
        A separator to use when splitting a delimited string into multiple numerical values
        as a vector. Only applicable to CSV input. Example: for a separator `'|'` the CSV
        value `1|2|3` would be transformed to a vector, `[1, 2, 3]`. When `None` the expected
        input format is an array of numerical values.

    """

    def __init__(self, config: Mapping):
        super().__init__(config)
        self.separator = self._transformation_kwargs.get("separator", None)

        self._sanity_check()
