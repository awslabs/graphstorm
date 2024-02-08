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

from graphstorm_processing.constants import HUGGINGFACE_TOKENIZE, HUGGINGFACE_EMB
from .feature_config_base import FeatureConfig


class HFConfig(FeatureConfig):
    """Feature configuration for huggingface text features.

    Supported kwargs
    ----------------
    action: str, required
        The type of huggingface action to use. Valid values are ["tokenize_hf", "embedding_hf"].
    hf_model: str, required
        The name of the huggingface lm model.
    max_seq_length: int, required
        The maximal length of the tokenization results.
    """

    def __init__(self, config: Mapping):
        super().__init__(config)
        self.action = self._transformation_kwargs.get("action")
        self.hf_model = self._transformation_kwargs.get("hf_model")
        self.max_seq_length = self._transformation_kwargs.get("max_seq_length")

        self._sanity_check()

    def _sanity_check(self) -> None:
        super()._sanity_check()
        assert self.action in [
            HUGGINGFACE_TOKENIZE,
            HUGGINGFACE_EMB,
        ], f"huggingface action needs to be one of {HUGGINGFACE_TOKENIZE, HUGGINGFACE_EMB}"
        assert isinstance(
            self.hf_model, str
        ), f"Expect hf_model to be a string, but got {self.hf_model}"
        assert (
            isinstance(self.max_seq_length, int) and self.max_seq_length > 0
        ), f"Expect max_seq_length {self.max_seq_length} be an integer and larger than zero."
