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

Base class for graph data processing.
"""
from abc import ABC
from typing import Dict, List
import os
import abc

from graphstorm_processing.config.config_parser import StructureConfig
from graphstorm_processing.config.feature_config_base import FeatureConfig


class GraphLoader(ABC):
    """Graph Loader base class

    Parameters
    ----------
    data_path : str
        Local path to input configuration file.
    local_metadata_path : str
        Output path for local metadata files.
    data_configs : Dict[str, List[StructureConfig]]
        Dictionary of graph structure configurations.
    """

    def __init__(
        self,
        data_path: str,
        local_metadata_path: str,
        data_configs: Dict[str, List[StructureConfig]],
    ):
        self._data_path = data_path
        self._output_path = local_metadata_path
        self._data_configs = data_configs
        self._feats: List[FeatureConfig] = []

        if not os.path.exists(local_metadata_path) and not local_metadata_path.startswith("s3://"):
            os.makedirs(local_metadata_path)

    @abc.abstractmethod
    def load(self) -> None:
        """
        Performs the graph loading, reading input files from storage, processing graph
        structure and node/edge features and writes processed output to storage.
        """

    @property
    def output_path(self):
        """
        Returns output path for local metadata files.
        """
        return self._output_path

    @property
    def features(self) -> List[FeatureConfig]:
        """
        Returns list of feature configurations.
        """
        return self._feats
