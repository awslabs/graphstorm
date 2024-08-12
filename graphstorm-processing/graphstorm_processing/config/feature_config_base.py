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

import abc
from typing import Any, Mapping, Optional, Sequence

from graphstorm_processing.constants import VALID_OUTDTYPE, TYPE_FLOAT32

from .data_config_base import DataStorageConfig


class FeatureConfig(abc.ABC):
    """
    Base class for feature configurations
    """

    def __init__(self, config: Mapping):
        self._config = config

        # TODO: Clarify if and how we plan to support multi-col processing for single feature type
        combined_cols = config["column"]
        if not isinstance(combined_cols, list):
            combined_cols = [combined_cols]
        self._cols: Sequence[str] = combined_cols
        self._transformation_name: str = config["transformation"]["name"]
        self._feat_name: str = config.get("name", config["column"])
        self._transformation_kwargs: Mapping[str, Any] = config["transformation"].get("kwargs", {})

        self._data_config = None
        if "data" in config:
            self._data_config = DataStorageConfig(**config["data"])

    def __str__(self) -> str:
        return str(self._config)

    @property
    def feat_name(self):
        """
        The name of the feature.
        """
        return self._feat_name

    @property
    def feat_type(self):
        """
        The type of transformation applied to the feature column.
        """
        return self._transformation_name

    @property
    def cols(self):
        """
        The column(s) that the feature configuration refers to.
        """
        return self._cols

    @property
    def transformation_kwargs(self) -> Mapping[str, Any]:
        """
        a dict of keyword arguments for the feature.
        """
        return self._transformation_kwargs

    def _sanity_check(self) -> None:
        """
        Checks that the configuration is valid
        """
        assert isinstance(self._transformation_name, str)
        assert isinstance(self._cols, list)
        assert isinstance(self._transformation_kwargs, dict)
        assert isinstance(self._feat_name, str)


class NoopFeatureConfig(FeatureConfig):
    """Feature configuration for features that do not need to be transformed.

    Supported kwargs
    ----------------
    out_dtype: str
        Output feature dtype. Currently, we support ``float32`` and ``float64``.
        Default is ``float32``
    separator: str
        When provided will treat the input as strings, split each value in the string using
        the separator, and convert the resulting list of floats into a float-vector feature.
    truncate_dim: int
        When provided, will truncate the output float-vector feature to the specified dimension.
        This is useful when the feature is a multi-dimensional vector and we only need
        a subset of the dimensions, e.g. for Matryoshka Representation Learning embeddings.
    """

    def __init__(self, config: Mapping):
        super().__init__(config)

        self.out_dtype: str = self._transformation_kwargs.get("out_dtype", TYPE_FLOAT32)
        self.value_separator: Optional[str] = self._transformation_kwargs.get("separator", None)
        self.truncate_dim: Optional[int] = self._transformation_kwargs.get("truncate_dim", None)

        self._sanity_check()

    def _sanity_check(self) -> None:
        super()._sanity_check()
        if self._data_config and self.value_separator and self._data_config.format != "csv":
            raise RuntimeError("separator should only be provided for CSV data")
        assert (
            self.out_dtype in VALID_OUTDTYPE
        ), f"Unsupported output dtype, expected one of {VALID_OUTDTYPE}, got {self.out_dtype}"
        assert self.truncate_dim is None or isinstance(
            self.truncate_dim, int
        ), f"truncate_dim should be an int or None, got {type(self.truncate_dim)}"
