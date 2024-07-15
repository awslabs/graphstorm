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

Configuration parsing for edges and nodes
"""

from abc import ABC
from typing import Any, Dict, List, Optional, Sequence

from graphstorm_processing.constants import SUPPORTED_FILE_TYPES
from .label_config_base import LabelConfig, EdgeLabelConfig, NodeLabelConfig
from .feature_config_base import FeatureConfig, NoopFeatureConfig
from .numerical_configs import (
    BucketNumericalFeatureConfig,
    MultiNumericalFeatureConfig,
    NumericalFeatureConfig,
)
from .categorical_configs import MultiCategoricalFeatureConfig
from .hf_configs import HFConfig
from .data_config_base import DataStorageConfig


def parse_feat_config(feature_dict: Dict) -> FeatureConfig:
    """Parses a feature configuration, converting it to the correct type of feature config.

    Parameters
    ----------
    feature_dict : Dict
        A feature configuration dictionary in GSProcessing format

    Returns
    -------
    FeatureConfig
        A feature configuration object

    Raises
    ------
    RuntimeError
        _description_
    """
    # Special case: missing transformation is parsed as a no-op
    if "transformation" not in feature_dict:
        feature_dict["transformation"] = {"name": "no-op"}
        return NoopFeatureConfig(feature_dict)

    transformation_name = feature_dict["transformation"]["name"]

    if transformation_name == "no-op":
        return NoopFeatureConfig(feature_dict)
    elif transformation_name == "numerical":
        return NumericalFeatureConfig(feature_dict)
    elif transformation_name == "multi-numerical":
        return MultiNumericalFeatureConfig(feature_dict)
    elif transformation_name == "bucket-numerical":
        return BucketNumericalFeatureConfig(feature_dict)
    elif transformation_name == "categorical":
        return FeatureConfig(feature_dict)
    elif transformation_name == "multi-categorical":
        return MultiCategoricalFeatureConfig(feature_dict)
    elif transformation_name == "huggingface":
        return HFConfig(feature_dict)
    else:
        raise RuntimeError(f"Unknown transformation name: '{transformation_name}'")


class StructureConfig(ABC):
    """
    Base class for edge and node configurations
    """

    def __init__(self, data_dict: Dict[str, Any]):
        self._data_config = DataStorageConfig(
            format=data_dict["format"],
            files=data_dict["files"],
            separator=data_dict.get("separator", None),
        )
        self._files = self._data_config.files
        self._separator = self._data_config.separator
        self._format = self._data_config.format
        self._feature_configs: List[FeatureConfig] = []
        self._labels: List[LabelConfig] = []

    @property
    def data_config(self) -> DataStorageConfig:
        """
        The data configuration for the structure.
        """
        return self._data_config

    @property
    def files(self) -> Sequence[str]:
        """
        The list of the files to be parsed.
        """
        return self._files

    @property
    def separator(self) -> Optional[str]:
        """
        The separator of the files. None if the input files are in Parquet format.
        """
        return self._separator

    @property
    def format(self) -> str:
        """
        The format of the files.
        """
        return self._format

    @property
    def label_configs(self) -> Optional[List[LabelConfig]]:
        """
        The list of labels, if they exist, None otherwise.
        """
        return self._labels if self._labels else None

    @property
    def feature_configs(self) -> Optional[List[FeatureConfig]]:
        """
        The list of labels, if they exist, None otherwise.
        """
        return self._feature_configs if self._feature_configs else None

    def set_labels(self, labels: List[LabelConfig]) -> None:
        """
        Set the list of labels to the provided value.
        """
        # TODO: Revisit how much we need this.
        self._labels = labels

    def sanity_check(self) -> None:
        """
        Perform sanity checks on the configuration input.
        """
        assert isinstance(self._files, list)
        assert isinstance(self._separator, (str, type(None)))
        assert isinstance(self._format, str)
        assert self.format in SUPPORTED_FILE_TYPES
        assert isinstance(self._feature_configs, list)
        assert isinstance(self._labels, list)


class EdgeConfig(StructureConfig):
    """Parsing edge data

    Parameters
    ----------
    edge_dict: dict
        Edge type configuration dictionary.
    data_dict: dict
        Data configuration for the edge type.

    Notes
    -----
    The edge dict must include the following keys:
    {
        "source" : {"column": "~from", "type": "movie"},
        "relation" : {"type": "included_in"},
        "dest" : {"column": "~to", "type": "genre"}
        "features" : [...], # optional
        "labels" : [...]  # optional
    }

    The data dict must be in the following format:
    {
        "format": "csv",
        "files": ["edges/user-rated-movie.csv"],
        "separator" : ","
    }

    """

    def __init__(self, edge_dict: Dict[str, Dict], data_dict: Dict[str, Any]):
        super().__init__(data_dict)
        self._src_col = edge_dict["source"]["column"]
        self._src_ntype = edge_dict["source"]["type"]
        self._dst_col = edge_dict["dest"]["column"]
        self._dst_ntype = edge_dict["dest"]["type"]
        self._rel_type = edge_dict["relation"]["type"]
        self._rel_col: Optional[str] = edge_dict["relation"].get("column", None)

        if "features" in edge_dict:
            for feature_dict in edge_dict["features"]:
                feat_config = parse_feat_config(feature_dict)
                self._feature_configs.append(feat_config)

        if "labels" in edge_dict:
            for label_config in edge_dict["labels"]:
                label = EdgeLabelConfig(label_config)
                self._labels.append(label)

        self.sanity_check()

    def get_relation_name(self) -> str:
        """
        Get the relation name.
        If only an edge type is defined, the relation name will be the edge type.

        If an edge relation column is also specified, the relation
        name will be edge_type-edge_relation_column.
        """
        if self._rel_col is None or self._rel_col == "":
            return self._rel_type
        else:
            return f"{self._rel_col}-{self._rel_type}"

    def sanity_check(self) -> None:
        super().sanity_check()
        assert isinstance(self._src_col, str)
        assert isinstance(self._src_ntype, str)
        assert isinstance(self._dst_col, str)
        assert isinstance(self._dst_ntype, str)
        assert isinstance(self._rel_type, str)
        assert isinstance(self._rel_col, (str, type(None)))

    @property
    def src_ntype(self) -> str:
        """
        The source node type
        """
        return self._src_ntype

    @property
    def dst_ntype(self) -> str:
        """
        The destination node type
        """
        return self._dst_ntype

    @property
    def rel_type(self) -> str:
        """
        The relation type prefix name.
        """
        return self._rel_type

    @property
    def src_col(self) -> str:
        """
        The source node column name
        """
        return self._src_col

    @property
    def dst_col(self) -> str:
        """
        The destination node column name
        """
        return self._dst_col

    @property
    def rel_col(self) -> Optional[str]:
        """
        The relation column. None if the edge is of a singular type.
        """
        return self._rel_col


class NodeConfig(StructureConfig):
    """Parsing node data

    Parameters
    ----------
    node_config: dict
        File config
    data_config: dict
        Data configuration of the input.

    Notes
    -----
    The config should in the following format:
    [
        {
            "data": {
                "format": "csv",
                "files": ["nodes/user.csv"],
                "separator" : ","
            },
            "column" : "~id",
            "type": "user",
            "labels" : [
                {
                  "column": "gender",
                  "type": "classification",
                  "split_rate" : {
                    "train": 0.8,
                    "val": 0.1,
                    "test": 0.1
                  }
                }
            ],
            "features": [
                {
                    "column": "age",
                    "transformation": {
                        "name": "no-op"
                    }
                }
            ]
        }
    ]
    """

    def __init__(self, node_config: Dict, data_config: Dict[str, Any]):
        # TODO: Since the data_config is part of the node_config, remove the arg
        super().__init__(data_config)
        self._node_col = node_config["column"]
        self._ntype = node_config["type"]
        self._node_config_dict = node_config
        if "features" in node_config:
            for feature_dict in node_config["features"]:
                feat_config = parse_feat_config(feature_dict)
                self._feature_configs.append(feat_config)
        if "labels" in node_config:
            for label_config in node_config["labels"]:
                label = NodeLabelConfig(label_config)
                self._labels.append(label)

        self.sanity_check()

    def sanity_check(self) -> None:
        super().sanity_check()
        assert isinstance(self._node_col, str)
        assert isinstance(self._ntype, str)
        assert isinstance(self._node_config_dict, dict)

    def __str__(self) -> str:
        return self._node_config_dict.__str__()

    @property
    def ntype(self):
        """
        The node type name
        """
        return self._ntype

    @property
    def node_col(self):
        """
        The column name that contains the node ids.
        """
        return self._node_col


def create_config_objects(graph_config: Dict[str, Any]) -> Dict[str, Sequence[StructureConfig]]:
    """Parses a GSProcessing JSON configuration dictionary and converts it to configuration objects.

    The input dict's expected format is:

    {
        "version" : "v1.0",
        "graph" : {
            "edges" : [
                {
                    "data": {
                        "format": "csv",
                        "files": ["edges/movie-included_in-genre.csv"],
                        "separator" : ","
                    },
                    "source" : {"column": "~from", "type": "movie"},
                    "relation" : {"type": "included_in"},
                    "dest" : {"column": "~to", "type": "genre"}
                },...
                ],
            "nodes" : [
                {
                    "data": {
                        "format": "csv",
                        "files": ["nodes/genre.csv"],
                        "separator" : ","
                    },
                    "column" : "~id",
                    "type": "genre"
                },...
                ]
        }
    }

    Parameters
    ----------
    graph_config : Dict[str, Any]
        A dictionary with a specific structure, describing stored graph data.

    Returns
    -------
    Dict[str, Sequence[StructureConfig]]
        Dictionary of node and edge configurations objects.
    """
    data_configs: Dict[str, Sequence[StructureConfig]] = {}
    edge_configs: List[EdgeConfig] = []
    node_configs: List[NodeConfig] = []
    # parse edges
    for edge_conf in graph_config["edges"]:
        edge_configs.append(EdgeConfig(edge_conf, edge_conf["data"]))
    # parse nodes
    if "nodes" in graph_config:
        for node_conf in graph_config["nodes"]:
            node_configs.append(NodeConfig(node_conf, node_conf["data"]))

    data_configs["edges"] = edge_configs
    data_configs["nodes"] = node_configs

    return data_configs
