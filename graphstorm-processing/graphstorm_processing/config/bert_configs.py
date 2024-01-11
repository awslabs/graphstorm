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
import numbers

from graphstorm_processing.constants import VALID_BERT_MODEL
from .feature_config_base import FeatureConfig


class BertConfig(FeatureConfig):
    """Feature configuration for single-column numerical features.

    Supported kwargs
    ----------------

    """

    def __init__(self, config: Mapping):
        super().__init__(config)
        self.bert_model = self._transformation_kwargs.get("bert_model", "none")
        self.max_seq_length = self._transformation_kwargs.get("max_seq_length", "none")

        self._sanity_check()

    def _sanity_check(self) -> None:
        super()._sanity_check()
        assert (
            self.bert_model in VALID_BERT_MODEL
        ), f"Unknown imputer requested, expected one of {VALID_BERT_MODEL}, got {self.bert_model}"
        assert isinstance(self.max_seq_length, int) and self.max_seq_length > 0, \
            f"Expect max_seq_length {self.max_seq_length} be an integer and larger than zero."

