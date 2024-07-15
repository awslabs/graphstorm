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
from .feature_config_base import FeatureConfig


class MultiCategoricalFeatureConfig(FeatureConfig):
    """Feature configuration for multi-column categorical features.

    Supported kwargs
    ----------------
    separator: str, optional
        A separator to use when splitting a delimited string into multiple numerical values
        as a vector. Only applicable to CSV input. Example: for a separator `'|'` the CSV
        value `1|2|3` would be transformed to a vector, `[1, 2, 3]`. When `None` the expected
        input format is an array of string values.

    """

    def __init__(self, config: Mapping):
        super().__init__(config)
        self.separator = self._transformation_kwargs.get("separator", None)

        self._sanity_check()
